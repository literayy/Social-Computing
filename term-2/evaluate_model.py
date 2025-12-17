import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import DebertaV2Tokenizer, DebertaV2Model

# ==================== 同步训练代码的模型结构 ====================
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
    
    def set_alpha(self, alpha):
        self.alpha = alpha

# 关键修复：同步训练代码的模型结构（增强版分类器）
class DomainAdaptiveDeBERTa(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.deberta = DebertaV2Model.from_pretrained(config.DEBERTA_PATH)
        self.hidden_size = self.deberta.config.hidden_size
        print(f"✓ DeBERTa hidden_size: {self.hidden_size}")
        
        # 初始化GRL（默认alpha降低，避免梯度反转过强）
        self.grl = GradientReversalLayer(alpha=getattr(config, 'GRL_ALPHA', 0.3))
        
        # 与训练代码一致的域分类器（增强版）
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, getattr(config, 'NUM_DOMAINS', 2))
        )
        
        # Bot分类器保持一致
        self.bot_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, getattr(config, 'NUM_BOT_CLASSES', 2))
        )
        
        # 与训练代码一致的性别分类器（增强版）
        self.gender_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, getattr(config, 'NUM_GENDER_CLASSES', 3))
        )
    
    def forward(self, input_ids, attention_mask, domain):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state[:, 0, :]
        
        reversed_features = self.grl(features)
        domain_logits = self.domain_classifier(reversed_features)
        
        bot_logits = self.bot_classifier(features)
        gender_logits = self.gender_classifier(features)
        
        return {
            'features': features,
            'domain_logits': domain_logits,
            'bot_logits': bot_logits,
            'gender_logits': gender_logits
        }

class DualDomainDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length, config):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.config = config
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        domain_label = 0 if row['domain'] == 'cresci' else 1
        task_label = int(row['label'])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'domain_label': torch.tensor(domain_label, dtype=torch.long),
            'task_label': torch.tensor(task_label, dtype=torch.long),
            'domain': row['domain']
        }

# 分层抽样函数
def stratified_sample_df(df, target_col, sample_size, random_state=42):
    """
    对DataFrame进行分层抽样，保证每个类别的样本数量平衡
    :param df: 原始DataFrame
    :param target_col: 分层依据的列名（label）
    :param sample_size: 总抽样数量
    :param random_state: 随机种子
    :return: 分层抽样后的DataFrame
    """
    if len(df) <= sample_size:
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # 计算每个类别应抽取的样本数
    class_counts = df[target_col].value_counts()
    num_classes = len(class_counts)
    
    # 每个类别至少抽取的样本数
    samples_per_class = min(class_counts.min(), sample_size // num_classes)
    
    # 对每个类别进行抽样
    sampled_dfs = []
    remaining = sample_size
    
    for class_label in class_counts.index:
        class_df = df[df[target_col] == class_label]
        sample_num = min(len(class_df), samples_per_class)
        sampled_class = class_df.sample(n=sample_num, random_state=random_state)
        sampled_dfs.append(sampled_class)
        remaining -= sample_num
    
    # 补充剩余的样本（如果还有）
    if remaining > 0:
        # 收集已抽样的索引
        sampled_indices = [idx for df in sampled_dfs for idx in df.index]
        remaining_df = df[~df.index.isin(sampled_indices)]
        # 从剩余样本中随机抽取
        supplement = remaining_df.sample(n=remaining, random_state=random_state)
        sampled_dfs.append(supplement)
    
    # 合并并打乱
    final_df = pd.concat(sampled_dfs, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return final_df

# 核心验证函数
def validate_saved_model():
    # 1. 加载配置
    try:
        from config import Config
        config = Config()
    except ImportError:
        # 兼容无config文件的情况，创建默认配置
        class DefaultConfig:
            DEBERTA_PATH = "microsoft/deberta-v2-xlarge"  # 根据实际使用的模型修改
            PREPROCESSED_CRESCI = ""
            PREPROCESSED_GENDER = ""
            OUTPUT_PATH = "./output"
            RANDOM_SEED = 42
            MAX_LENGTH = 128
            MIN_TEXT_LENGTH = 10
            BATCH_SIZE = 16
            MAX_TEST_SAMPLES = 500
            LAMBDA_TASK = 1.0
            LAMBDA_DOMAIN = 0.3
            NUM_DOMAINS = 2
            NUM_BOT_CLASSES = 2
            NUM_GENDER_CLASSES = 3
            GRL_ALPHA = 0.3
        config = DefaultConfig()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 2. 加载Tokenizer和模型
    tokenizer = DebertaV2Tokenizer.from_pretrained(config.DEBERTA_PATH)
    model = DomainAdaptiveDeBERTa(config).to(device)
    
    # 3. 加载已保存的模型权重（增加容错逻辑）
    model_path = r"F:\social-compute\output\models\best_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 修复权重加载：忽略不匹配的层（兼容模型结构微调）
    def load_state_dict_with_adjustment(model, state_dict):
        model_dict = model.state_dict()
        # 过滤掉不匹配的层
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                print(f"⚠️  跳过不匹配的权重: {k} (模型shape: {model_dict.get(k, '不存在').shape if k in model_dict else '不存在'}, 权重shape: {v.shape})")
        
        # 更新模型权重
        model_dict.update(filtered_state_dict)
        model.load_state_dict(model_dict)
        print(f"✓ 成功加载 {len(filtered_state_dict)}/{len(state_dict)} 个匹配的权重参数")
    
    load_state_dict_with_adjustment(model, state_dict)
    print(f"✓ 成功加载模型: {model_path}")
    
    # 4. 加载测试数据
    print("\n=== 加载并处理独立验证集 ===")
    test_set_path = os.path.join(config.OUTPUT_PATH, "test_set.csv")
    full_test_df = None
    
    if os.path.exists(test_set_path):
        # 加载全量测试集
        full_test_df = pd.read_csv(test_set_path)
        # 过滤短文本（和训练逻辑一致）
        full_test_df = full_test_df[full_test_df['text'].str.len() >= config.MIN_TEXT_LENGTH].reset_index(drop=True)
        
        print(f"原始测试集统计:")
        print(f"  总样本数: {len(full_test_df)}")
        print(f"  域分布: {full_test_df['domain'].value_counts().to_dict()}")
        
        # 拆分Cresci和Gender域
        cresci_df = full_test_df[full_test_df['domain'] == 'cresci'].reset_index(drop=True)
        gender_df = full_test_df[full_test_df['domain'] == 'gender'].reset_index(drop=True)
        
        print(f"  Cresci域标签分布: {cresci_df['label'].value_counts().to_dict() if len(cresci_df) > 0 else '空'}")
        print(f"  Gender域标签分布: {gender_df['label'].value_counts().to_dict() if len(gender_df) > 0 else '空'}")
        
        # 确定抽样大小
        max_test_samples = getattr(config, 'MAX_TEST_SAMPLES', 500)
        samples_per_domain = max_test_samples // 2
        
        # 分层抽样
        cresci_sampled = stratified_sample_df(cresci_df, 'label', samples_per_domain, config.RANDOM_SEED) if len(cresci_df) > 0 else pd.DataFrame()
        gender_sampled = stratified_sample_df(gender_df, 'label', samples_per_domain, config.RANDOM_SEED) if len(gender_df) > 0 else pd.DataFrame()
        
        # 合并抽样后的数据集
        test_dfs = []
        if not cresci_sampled.empty:
            test_dfs.append(cresci_sampled)
        if not gender_sampled.empty:
            test_dfs.append(gender_sampled)
        
        if test_dfs:
            test_df = pd.concat(test_dfs, ignore_index=True)
            test_df = test_df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
        else:
            raise ValueError("测试集为空，请检查数据路径")
        
        print(f"分层抽样后测试集统计:")
        print(f"  总样本数: {len(test_df)}")
        print(f"  Cresci域标签分布: {test_df[test_df['domain'] == 'cresci']['label'].value_counts().to_dict() if 'domain' in test_df.columns else '空'}")
        print(f"  Gender域标签分布: {test_df[test_df['domain'] == 'gender']['label'].value_counts().to_dict() if 'domain' in test_df.columns else '空'}")
        
    else:
        # 兼容原有逻辑
        print("⚠️  未找到测试集，使用配置文件中的原始数据路径")
        cresci_df = pd.read_csv(config.PREPROCESSED_CRESCI) if config.PREPROCESSED_CRESCI else pd.DataFrame()
        gender_df = pd.read_csv(config.PREPROCESSED_GENDER) if config.PREPROCESSED_GENDER else pd.DataFrame()
        
        cresci_df = cresci_df[cresci_df['text'].str.len() >= config.MIN_TEXT_LENGTH].reset_index(drop=True)
        gender_df = gender_df[gender_df['text'].str.len() >= config.MIN_TEXT_LENGTH].reset_index(drop=True)
        
        def split_data(df, train_ratio=0.7, val_ratio=0.15):
            if len(df) == 0:
                return pd.DataFrame()
            df = df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
            n = len(df)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            return df[val_end:].reset_index(drop=True)
        
        cresci_test = split_data(cresci_df)
        gender_test = split_data(gender_df)
        
        max_test_samples = getattr(config, 'MAX_TEST_SAMPLES', 500)
        samples_per_domain = max_test_samples // 2
        
        if len(cresci_test) > samples_per_domain:
            cresci_test = stratified_sample_df(cresci_test, 'label', samples_per_domain, config.RANDOM_SEED)
        if len(gender_test) > samples_per_domain:
            gender_test = stratified_sample_df(gender_test, 'label', samples_per_domain, config.RANDOM_SEED)
            
        test_df = pd.concat([cresci_test, gender_test], ignore_index=True)
        test_df = test_df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)

    # 5. 创建测试集DataLoader
    test_dataset = DualDomainDataset(test_df, tokenizer, config.MAX_LENGTH, config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )
    
    # 6. 模型评估
    print("\n=== 开始评估模型 ===")
    model.eval()
    criterion_domain = nn.CrossEntropyLoss()
    criterion_task = nn.CrossEntropyLoss()
    
    total_loss = 0
    all_task_preds = []
    all_task_labels = []
    all_domain_preds = []
    all_domain_labels = []
    cresci_preds = []
    cresci_labels = []
    gender_preds = []
    gender_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            domain_labels = batch['domain_label'].to(device)
            task_labels = batch['task_label'].to(device)
            domains = batch['domain']
            
            outputs = model(input_ids, attention_mask, domains)
            
            # 计算损失
            domain_loss = criterion_domain(outputs['domain_logits'], domain_labels)
            cresci_mask = torch.tensor([d == 'cresci' for d in domains]).to(device)
            gender_mask = ~cresci_mask
            
            task_loss = 0
            if cresci_mask.any():
                bot_loss = criterion_task(outputs['bot_logits'][cresci_mask], task_labels[cresci_mask])
                task_loss += bot_loss
                bot_preds = outputs['bot_logits'][cresci_mask].argmax(dim=1)
                cresci_preds.extend(bot_preds.cpu().numpy())
                cresci_labels.extend(task_labels[cresci_mask].cpu().numpy())
            
            if gender_mask.any():
                gender_loss = criterion_task(outputs['gender_logits'][gender_mask], task_labels[gender_mask])
                task_loss += gender_loss
                g_preds = outputs['gender_logits'][gender_mask].argmax(dim=1)
                gender_preds.extend(g_preds.cpu().numpy())
                gender_labels.extend(task_labels[gender_mask].cpu().numpy())
            
            loss = (config.LAMBDA_TASK * task_loss + config.LAMBDA_DOMAIN * domain_loss)
            total_loss += loss.item()
            
            # 收集预测结果
            domain_preds = outputs['domain_logits'].argmax(dim=1)
            all_domain_preds.extend(domain_preds.cpu().numpy())
            all_domain_labels.extend(domain_labels.cpu().numpy())
            
            for i, domain in enumerate(domains):
                if domain == 'cresci':
                    pred = outputs['bot_logits'][i].argmax()
                else:
                    pred = outputs['gender_logits'][i].argmax()
                all_task_preds.append(pred.cpu().item())
                all_task_labels.append(task_labels[i].cpu().item())
            
            del input_ids, attention_mask, outputs, loss
            gc.collect()
    
    # 7. 计算评估指标
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    task_acc = accuracy_score(all_task_labels, all_task_preds) if all_task_labels else 0
    domain_acc = accuracy_score(all_domain_labels, all_domain_preds) if all_domain_labels else 0
    bot_acc = accuracy_score(cresci_labels, cresci_preds) if cresci_labels else 0
    gender_acc = accuracy_score(gender_labels, gender_preds) if gender_labels else 0
    
    # 8. 输出详细结果
    print("\n" + "="*60)
    print("模型测试结果汇总（分层抽样后）")
    print("="*60)
    print(f"测试集平均损失: {avg_loss:.4f}")
    print(f"整体任务准确率: {task_acc:.4f} ({task_acc*100:.2f}%)")
    print(f"域判别准确率: {domain_acc:.4f} ({domain_acc*100:.2f}%)")
    print(f"Bot检测准确率: {bot_acc:.4f} ({bot_acc*100:.2f}%)")
    print(f"性别分类准确率: {gender_acc:.4f} ({gender_acc*100:.2f}%)")
    print("="*60)
    
    # 9. 生成分类报告
    print("\n=== 详细分类报告（分层抽样后）===")
    # Bot检测分类报告
    if cresci_labels:
        print("\n【Bot检测（Cresci域）】")
        print(f"样本分布: Human={sum(1 for l in cresci_labels if l == 0)}, Bot={sum(1 for l in cresci_labels if l == 1)}")
        print(classification_report(
            cresci_labels, 
            cresci_preds, 
            target_names=['Human (0)', 'Bot (1)'],
            zero_division=0
        ))
    
    # 性别分类报告
    if gender_labels:
        print("\n【性别分类（Gender域）】")
        print(f"样本分布: Male={sum(1 for l in gender_labels if l == 0)}, Female={sum(1 for l in gender_labels if l == 1)}, Brand={sum(1 for l in gender_labels if l == 2)}")
        print(classification_report(
            gender_labels, 
            gender_preds, 
            target_names=['Male (0)', 'Female (1)', 'Brand (2)'],
            zero_division=0
        ))
    
    # 10. 保存结果到文件
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    result_path = os.path.join(config.OUTPUT_PATH, "model_validation_result_stratified.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("模型验证结果（分层抽样后）\n")
        f.write("="*60 + "\n")
        f.write(f"测试集平均损失: {avg_loss:.4f}\n")
        f.write(f"整体任务准确率: {task_acc:.4f} ({task_acc*100:.2f}%)\n")
        f.write(f"域判别准确率: {domain_acc:.4f} ({domain_acc*100:.2f}%)\n")
        f.write(f"Bot检测准确率: {bot_acc:.4f} ({bot_acc*100:.2f}%)\n")
        f.write(f"性别分类准确率: {gender_acc:.4f} ({gender_acc*100:.2f}%)\n")
        
        f.write("\n【抽样信息】\n")
        f.write(f"原始测试集大小: {len(full_test_df) if full_test_df is not None else 'N/A'}\n")
        f.write(f"分层抽样后大小: {len(test_df)}\n")
        if 'cresci_df' in locals() and len(cresci_df) > 0:
            f.write(f"Cresci域原始分布: {cresci_df['label'].value_counts().to_dict()}\n")
            f.write(f"Cresci域抽样分布: {test_df[test_df['domain'] == 'cresci']['label'].value_counts().to_dict()}\n")
        if 'gender_df' in locals() and len(gender_df) > 0:
            f.write(f"Gender域原始分布: {gender_df['label'].value_counts().to_dict()}\n")
            f.write(f"Gender域抽样分布: {test_df[test_df['domain'] == 'gender']['label'].value_counts().to_dict()}\n")
        
        if cresci_labels:
            f.write("\n【Bot检测分类报告】\n")
            f.write(classification_report(cresci_labels, cresci_preds, target_names=['Human (0)', 'Bot (1)'], zero_division=0))
        
        if gender_labels:
            f.write("\n【性别分类报告】\n")
            f.write(classification_report(gender_labels, gender_preds, target_names=['Male (0)', 'Female (1)', 'Brand (2)'], zero_division=0))
    
    print(f"\n✓ 分层抽样验证结果已保存至: {result_path}")
    
    # 11. 绘制混淆矩阵
    # Bot检测混淆矩阵
    if cresci_labels:
        cm_bot = confusion_matrix(cresci_labels, cresci_preds)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm_bot, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Bot Detection Confusion Matrix (Total: {len(cresci_labels)})')
        plt.colorbar()
        plt.xticks([0, 1], ['Human', 'Bot'])
        plt.yticks([0, 1], ['Human', 'Bot'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 标注数值和比例
        for i in range(cm_bot.shape[0]):
            for j in range(cm_bot.shape[1]):
                percentage = cm_bot[i, j] / sum(cm_bot[i]) * 100 if sum(cm_bot[i]) > 0 else 0
                plt.text(j, i, f"{cm_bot[i, j]}\n({percentage:.1f}%)", 
                         ha='center', va='center', 
                         color='white' if cm_bot[i, j] > cm_bot.max()/2 else 'black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_PATH, 'bot_confusion_matrix_stratified.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 性别分类混淆矩阵
    if gender_labels:
        cm_gender = confusion_matrix(gender_labels, gender_preds)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm_gender, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Gender Classification Confusion Matrix (Total: {len(gender_labels)})')
        plt.colorbar()
        plt.xticks([0, 1, 2], ['Male', 'Female', 'Brand'])
        plt.yticks([0, 1, 2], ['Male', 'Female', 'Brand'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 标注数值和比例
        for i in range(cm_gender.shape[0]):
            for j in range(cm_gender.shape[1]):
                percentage = cm_gender[i, j] / sum(cm_gender[i]) * 100 if sum(cm_gender[i]) > 0 else 0
                plt.text(j, i, f"{cm_gender[i, j]}\n({percentage:.1f}%)", 
                         ha='center', va='center', 
                         color='white' if cm_gender[i, j] > cm_gender[i].max()/2 else 'black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_PATH, 'gender_confusion_matrix_stratified.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ 分层抽样混淆矩阵已保存至: {config.OUTPUT_PATH}")

if __name__ == "__main__":
    validate_saved_model()