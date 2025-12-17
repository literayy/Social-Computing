import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from transformers import DebertaV2Tokenizer, DebertaV2Model, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import os
import gc
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import random

# ==================== 梯度反转层 (优化：动态alpha + 稳定梯度) ====================
class GradientReversalFunction(Function):
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
        """新增：动态调整alpha值（核心优化）"""
        self.alpha = alpha

# ==================== 数据集类 (优化：性别分类样本平衡) ===================
class DualDomainDataset(Dataset):
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
            'domain': row['domain'],
            'idx': torch.tensor(idx, dtype=torch.long)
        }

# ==================== 域适应模型 (核心优化：增强分类器+稳定训练) ====================
class DomainAdaptiveDeBERTa(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.deberta = DebertaV2Model.from_pretrained(config.DEBERTA_PATH)
        self.hidden_size = self.deberta.config.hidden_size
        print(f"✓ DeBERTa hidden_size: {self.hidden_size}")
        
        # 初始化GRL（默认alpha降低，避免梯度反转过强）
        self.grl = GradientReversalLayer(alpha=config.GRL_ALPHA if hasattr(config, 'GRL_ALPHA') else 0.3)
        
        # 优化1：增强域分类器（解决域判别准确率过低）
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),  # 新增层归一化，稳定训练
            nn.ReLU(),
            nn.Dropout(0.3),    # 提高dropout，防止过拟合
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, config.NUM_DOMAINS)
        )
        
        # Bot分类器保持（原有效果好）
        self.bot_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, config.NUM_BOT_CLASSES)
        )
        
        # 优化2：增强性别分类器（解决准确率低）
        self.gender_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),  # 新增层归一化
            nn.ReLU(),
            nn.Dropout(0.3),    # 提高dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, config.NUM_GENDER_CLASSES)
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

# ==================== 标签平滑损失 (新增：减少过拟合) ====================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# ==================== 训练器 (核心优化：无原模型时从0训练) ====================
class DomainAdaptationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU总内存: {total_mem:.2f} GB")
        
        self._set_seed()
        self.tokenizer = self._load_tokenizer()
        self.model = self._build_model()
        
        # 加载已有best_model.pt（核心修改：无模型时跳过）
        self.original_model_path = r"F:\social-compute\output\models\best_model.pt"
        self._load_pretrained_model()  # 修改后的加载逻辑
        
        # 训练历史和最优指标
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_acc': [], 'domain_acc': [],  # 新增域准确率跟踪
            'bot_acc': [], 'gender_acc': []    # 新增细分任务准确率
        }
        self.best_val_acc = 0.0  
        self.sampled_indices = set()
        self.full_train_df = None
        self.val_df = None
        self.test_df = None

    def _set_seed(self):
        random.seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
        torch.manual_seed(self.config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.RANDOM_SEED)
            torch.cuda.manual_seed_all(self.config.RANDOM_SEED)

    def _load_tokenizer(self):
        try:
            tokenizer = DebertaV2Tokenizer.from_pretrained(self.config.DEBERTA_PATH)
            print("✓ Tokenizer加载成功")
            return tokenizer
        except Exception as e:
            raise Exception(f"Tokenizer加载失败: {e}")

    def _build_model(self):
        try:
            model = DomainAdaptiveDeBERTa(self.config).to(self.device)
            print("✓ 域适应模型构建成功")
            param_count = sum(p.numel() for p in model.parameters())
            print(f"模型参数量: {param_count:,} ({param_count/1e6:.2f}M)")
            return model
        except Exception as e:
            raise Exception(f"模型构建失败: {e}")
    
    def _load_pretrained_model(self):
        """修改：无原模型时跳过加载，从0开始训练"""
        if os.path.exists(self.original_model_path):
            try:
                checkpoint = torch.load(self.original_model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
                print(f"✓ 成功加载预训练模型: {self.original_model_path}")
                print(f"ℹ️  将基于已有模型进行增量训练")
            except Exception as e:
                print(f"⚠️  模型加载失败: {e}")
                print(f"ℹ️  放弃加载预训练模型，从0开始训练")
        else:
            print(f"⚠️  未找到原模型: {self.original_model_path}")
            print(f"ℹ️  从0开始训练新模型")

    def load_full_data(self):
        print("\n=== 加载全量预处理数据（按7:1.5:1.5划分训练/验证/测试集）===")
        # 加载原始数据
        cresci_df = pd.read_csv(self.config.PREPROCESSED_CRESCI)
        gender_df = pd.read_csv(self.config.PREPROCESSED_GENDER)

        # 数据清洗
        cresci_df = cresci_df[cresci_df['text'].str.len() >= self.config.MIN_TEXT_LENGTH].reset_index(drop=True)
        gender_df = gender_df[gender_df['text'].str.len() >= self.config.MIN_TEXT_LENGTH].reset_index(drop=True)

        # 定义划分函数（严格按7:1.5:1.5划分训练/验证/测试集）
        def split_data(df, train_ratio, val_ratio, test_ratio):
            assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
            df = df.sample(frac=1, random_state=self.config.RANDOM_SEED).reset_index(drop=True)
            n = len(df)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            train_df = df[:train_end].reset_index(drop=True)
            val_df = df[train_end:val_end].reset_index(drop=True)
            test_df = df[val_end:].reset_index(drop=True)
            return train_df, val_df, test_df

        # 按7:1.5:1.5划分Cresci和Gender数据集
        cresci_train, cresci_val, cresci_test = split_data(
            cresci_df, 
            self.config.TRAIN_RATIO, 
            self.config.VAL_RATIO, 
            self.config.TEST_RATIO
        )
        gender_train, gender_val, gender_test = split_data(
            gender_df, 
            self.config.TRAIN_RATIO, 
            self.config.VAL_RATIO, 
            self.config.TEST_RATIO
        )

        # 合并跨域数据集（训练/验证/测试完全独立）
        self.full_train_df = pd.concat([cresci_train, gender_train], ignore_index=True)
        val_df_raw = pd.concat([cresci_val, gender_val], ignore_index=True)
        self.test_df = pd.concat([cresci_test, gender_test], ignore_index=True)

        # 优化：性别分类分层抽样（保证Male/Female/Brand样本均衡）
        if len(val_df_raw) >= self.config.MAX_VAL_SAMPLES:
            # 先按域分层，再按性别标签分层
            val_df_cresci = val_df_raw[val_df_raw['domain'] == 'cresci'].sample(
                n=min(100, len(val_df_raw[val_df_raw['domain'] == 'cresci'])),
                random_state=self.config.RANDOM_SEED
            )
            # 性别域按标签分层抽样
            gender_val_grouped = val_df_raw[val_df_raw['domain'] == 'gender'].groupby('label')
            gender_samples = []
            for label, group in gender_val_grouped:
                sample_size = min(34, len(group))  # 3类各≈34条，总计≈100条
                gender_samples.append(group.sample(sample_size, random_state=self.config.RANDOM_SEED))
            val_df_gender = pd.concat(gender_samples, ignore_index=True)
            
            self.val_df = pd.concat([val_df_cresci, val_df_gender], ignore_index=True)
            # 补充到200条
            if len(self.val_df) < self.config.MAX_VAL_SAMPLES:
                remaining_val = val_df_raw[~val_df_raw.index.isin(self.val_df.index)]
                supplement = remaining_val.sample(n=self.config.MAX_VAL_SAMPLES - len(self.val_df), random_state=self.config.RANDOM_SEED)
                self.val_df = pd.concat([self.val_df, supplement], ignore_index=True).reset_index(drop=True)
        else:
            self.val_df = val_df_raw.reset_index(drop=True)
            print(f"⚠️  原始验证集仅{len(val_df_raw)}条，不足200条，使用全部作为验证集")

        # 给训练集加唯一索引
        self.full_train_df['unique_idx'] = range(len(self.full_train_df))

        # 保存验证集/测试集
        self.val_df.to_csv(os.path.join(self.config.OUTPUT_PATH, "val_set.csv"), index=False)
        self.test_df.to_csv(os.path.join(self.config.OUTPUT_PATH, "test_set.csv"), index=False)

        # 打印划分结果
        print(f"Cresci - 训练集: {len(cresci_train)} | 原始验证集: {len(cresci_val)} | 测试集: {len(cresci_test)}")
        print(f"Gender - 训练集: {len(gender_train)} | 原始验证集: {len(gender_val)} | 测试集: {len(gender_test)}")
        print(f"全量 - 训练集: {len(self.full_train_df)} | 验证集: {len(self.val_df)}  | 测试集: {len(self.test_df)}")
        # 打印性别分布
        if 'domain' in self.val_df.columns and 'label' in self.val_df.columns:
            gender_dist = self.val_df[self.val_df['domain'] == 'gender']['label'].value_counts()
            print(f"验证集性别分布: {gender_dist.to_dict()}")

    def sample_train_data(self):
        """动态采样（优化：性别分类样本平衡）"""
        all_indices = set(self.full_train_df['unique_idx'].tolist())
        remaining_indices = all_indices - self.sampled_indices
        
        if len(remaining_indices) < self.config.MAX_TRAIN_SAMPLES:
            print(f"\n⚠️  剩余未采样数据不足，重置采样记录（已覆盖全量数据）")
            self.sampled_indices = set()
            remaining_indices = all_indices
        
        # 优化：分层采样（保证性别分类各类样本均衡）
        def balanced_sample(df, sample_size, remaining_indices):
            df_remaining = df[df['unique_idx'].isin(remaining_indices)]
            
            # 先按域拆分
            cresci_df = df_remaining[df_remaining['domain'] == 'cresci']
            gender_df = df_remaining[df_remaining['domain'] == 'gender']
            
            # Cresci域：Bot/Human平衡
            cresci_samples = []
            cresci_groups = cresci_df.groupby('label')
            for name, group in cresci_groups:
                sample_num = min(len(group), sample_size//4)  # 占总样本的1/2
                cresci_samples.append(group.sample(sample_num, random_state=np.random.randint(1000)))
            cresci_sampled = pd.concat(cresci_samples, ignore_index=True)
            
            # Gender域：Male/Female/Brand平衡
            gender_samples = []
            gender_groups = gender_df.groupby('label')
            for name, group in gender_groups:
                sample_num = min(len(group), sample_size//6)  # 占总样本的1/2，3类均分
                gender_samples.append(group.sample(sample_num, random_state=np.random.randint(1000)))
            gender_sampled = pd.concat(gender_samples, ignore_index=True)
            
            # 合并并补充到指定大小
            sampled_df = pd.concat([cresci_sampled, gender_sampled], ignore_index=True)
            if len(sampled_df) < sample_size:
                remaining_df = df_remaining[~df_remaining['unique_idx'].isin(sampled_df['unique_idx'])]
                supplement = remaining_df.sample(sample_size - len(sampled_df), random_state=np.random.randint(1000))
                sampled_df = pd.concat([sampled_df, supplement], ignore_index=True)
            
            return sampled_df.sample(frac=1).head(sample_size)
        
        sampled_train_df = balanced_sample(self.full_train_df, self.config.MAX_TRAIN_SAMPLES, remaining_indices)
        sampled_idx = set(sampled_train_df['unique_idx'].tolist())
        self.sampled_indices.update(sampled_idx)
        
        train_dataset = DualDomainDataset(sampled_train_df, self.tokenizer, self.config.MAX_LENGTH, self.config)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
        
        print(f"\n🔄 动态采样完成: {len(sampled_train_df)} 条")
        print(f"   域分布: {sampled_train_df['domain'].value_counts().to_dict()}")
        if 'domain' in sampled_train_df.columns and 'label' in sampled_train_df.columns:
            gender_label_dist = sampled_train_df[sampled_train_df['domain'] == 'gender']['label'].value_counts()
            print(f"   性别标签分布: {gender_label_dist.to_dict()}")
        print(f"   已采样占比: {len(self.sampled_indices)}/{len(self.full_train_df)} ({len(self.sampled_indices)/len(self.full_train_df)*100:.1f}%)")
        return train_loader

    def _eval_current_model(self):
        """优化：详细评估（域准确率+Bot+性别分类单独评估）"""
        if self.val_df is None or len(self.val_df) == 0:
            raise ValueError("独立验证集未加载！请先调用load_full_data()")
        
        self.model.eval()
        total_correct = 0
        total = 0
        domain_correct = 0
        bot_correct, bot_total = 0, 0
        gender_correct, gender_total = 0, 0
        
        eval_dataset = DualDomainDataset(self.val_df, self.tokenizer, self.config.MAX_LENGTH, self.config)
        eval_loader = DataLoader(eval_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                task_labels = batch['task_label'].to(self.device)
                domain_labels = batch['domain_label'].to(self.device)
                domains = batch['domain']
                
                outputs = self.model(input_ids, attention_mask, domains)
                
                # 域判别准确率
                domain_preds = outputs['domain_logits'].argmax(dim=1)
                domain_correct += (domain_preds == domain_labels).sum().item()
                
                # 核心任务准确率
                for i, domain in enumerate(domains):
                    pred = outputs['bot_logits'][i].argmax() if domain == 'cresci' else outputs['gender_logits'][i].argmax()
                    total_correct += (pred == task_labels[i]).item()
                    
                    # Bot/性别分类单独统计
                    if domain == 'cresci':
                        bot_correct += (pred == task_labels[i]).item()
                        bot_total += 1
                    else:
                        gender_correct += (pred == task_labels[i]).item()
                        gender_total += 1
                total += len(domains)
        
        # 计算各类准确率
        current_acc = total_correct / total
        domain_acc = domain_correct / total if total > 0 else 0
        bot_acc = bot_correct / bot_total if bot_total > 0 else 0
        gender_acc = gender_correct / gender_total if gender_total > 0 else 0
        
        # 记录历史
        self.history['val_acc'].append(current_acc)
        self.history['domain_acc'].append(domain_acc)
        self.history['bot_acc'].append(bot_acc)
        self.history['gender_acc'].append(gender_acc)
        
        print(f"📊 评估详情 - 整体: {current_acc:.4f} | 域判别: {domain_acc:.4f} | Bot: {bot_acc:.4f} | 性别: {gender_acc:.4f}")
        self.model.train()
        return current_acc

    def _save_best_model(self, current_acc, epoch):
        """优化：保存更详细的信息"""
        is_best = False
        if current_acc > self.best_val_acc:
            is_best = True
            print(f"\n📈 验证集准确率提升: {self.best_val_acc:.4f} → {current_acc:.4f}，标记为最优模型")
            self.best_val_acc = current_acc
            best_model_path = self.original_model_path
        else:
            print(f"\n📉 验证集准确率未提升: {current_acc:.4f} ≤ {self.best_val_acc:.4f}")

        # 确保模型保存目录存在（新增：防止路径不存在报错）
        os.makedirs(os.path.dirname(self.original_model_path), exist_ok=True)
        
        # 保存当前轮次模型
        coverage = len(self.sampled_indices) / len(self.full_train_df) * 100
        epoch_model_path = os.path.join(
            os.path.dirname(self.original_model_path),
            f"model_epoch_{epoch+1}_val_{current_acc:.4f}_domain_{self.history['domain_acc'][-1]:.4f}_gender_{self.history['gender_acc'][-1]:.4f}.pt"
        )
        
        # 保存完整信息
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch+1,
            'val_acc': current_acc,
            'domain_acc': self.history['domain_acc'][-1],
            'bot_acc': self.history['bot_acc'][-1],
            'gender_acc': self.history['gender_acc'][-1],
            'train_acc': self.history['train_acc'][-1],
            'train_loss': self.history['train_loss'][-1],
            'data_coverage': coverage,
            'sampled_indices': list(self.sampled_indices),
            'config': self.config,
            'is_best': is_best
        }, epoch_model_path)
        print(f"💾 第{epoch+1}轮模型保存至: {epoch_model_path}")

        # 更新最优模型链接
        if is_best:
            if os.name == 'nt':
                # Windows下先删除原有文件（避免链接失败）
                if os.path.exists(best_model_path):
                    try:
                        os.remove(best_model_path)
                    except:
                        pass
                # 尝试创建硬链接
                try:
                    os.system(f'mklink /H "{best_model_path}" "{epoch_model_path}"')
                except:
                    # 链接失败则直接复制文件
                    import shutil
                    shutil.copyfile(epoch_model_path, best_model_path)
            else:
                if os.path.exists(best_model_path):
                    os.unlink(best_model_path)
                os.symlink(epoch_model_path, best_model_path)
            print(f"🔖 最优模型链接已更新为: {best_model_path}")

    def train(self):
        """核心训练逻辑：优化损失权重+动态alpha+标签平滑"""
        print("\n=== 模型训练（支持从0训练/增量训练）===")
        self.load_full_data()
        
        # CPU模式冻结层
        if self.device.type == 'cpu' and hasattr(self.config, 'FREEZE_LAYERS') and self.config.FREEZE_LAYERS > 0:
            print(f"⚠️  CPU模式: 冻结DeBERTa前{self.config.FREEZE_LAYERS}层")
            for name, param in self.model.deberta.named_parameters():
                if 'embeddings' in name or any(f'layer.{i}.' in name for i in range(self.config.FREEZE_LAYERS)):
                    param.requires_grad = False
        
        # 优化器（区分增量训练和从零训练的学习率）
        if hasattr(self, '_pretrained_loaded') and self._pretrained_loaded:
            # 增量训练：更低的学习率
            lr = self.config.LEARNING_RATE * 0.05
            print(f"ℹ️  增量训练模式 - 学习率: {lr}")
        else:
            # 从零训练：正常学习率
            lr = self.config.LEARNING_RATE
            print(f"ℹ️  从零训练模式 - 学习率: {lr}")
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=self.config.WEIGHT_DECAY if hasattr(self.config, 'WEIGHT_DECAY') else 0.005
        )
        
        total_steps = self.config.NUM_EPOCHS * (self.config.MAX_TRAIN_SAMPLES // self.config.BATCH_SIZE)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
        
        # 优化3：使用标签平滑损失（减少过拟合）
        criterion_domain = LabelSmoothingLoss(self.config.NUM_DOMAINS, smoothing=0.1)
        criterion_task = LabelSmoothingLoss(max(self.config.NUM_BOT_CLASSES, self.config.NUM_GENDER_CLASSES), smoothing=0.1)
        
        # 优化4：调整损失权重（降低域损失，提升性别分类权重）
        lambda_task = self.config.LAMBDA_TASK if hasattr(self.config, 'LAMBDA_TASK') else 1.0
        lambda_domain = self.config.LAMBDA_DOMAIN if hasattr(self.config, 'LAMBDA_DOMAIN') else 0.3  # 降低域损失权重
        gender_weight = 2.0  # 性别分类损失加权
        
        # 增量训练
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"{'='*60}")
            
            # 优化5：动态调整GRL的alpha值（线性增加）
            alpha = 0.1 + (0.8) * epoch / self.config.NUM_EPOCHS  # 从0.1增加到0.9
            self.model.grl.set_alpha(alpha)
            print(f"🔧 当前GRL alpha值: {alpha:.4f}")
            
            train_loader = self.sample_train_data()
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            start_time = time.time()
            
            pbar = tqdm(train_loader, desc=f"训练", disable=(self.device.type == 'cpu'))
            for batch_idx, batch in enumerate(pbar):
                try:
                    # CPU进度打印
                    if self.device.type == 'cpu' and batch_idx % self.config.LOG_INTERVAL == 0:
                        elapsed = time.time() - start_time if batch_idx > 0 else 0
                        eta = (elapsed/(batch_idx+1)) * (len(train_loader)-batch_idx-1) if batch_idx > 0 else 0
                        print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {train_loss/(batch_idx+1):.4f} | Acc: {train_correct/max(train_total,1):.4f} | ETA: {eta/60:.1f}min")
                    
                    # 前向/反向传播
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    domain_labels = batch['domain_label'].to(self.device)
                    task_labels = batch['task_label'].to(self.device)
                    domains = batch['domain']
                    
                    outputs = self.model(input_ids, attention_mask, domains)
                    domain_loss = criterion_domain(outputs['domain_logits'], domain_labels)
                    
                    cresci_mask = torch.tensor([d == 'cresci' for d in domains]).to(self.device)
                    gender_mask = ~cresci_mask
                    task_loss = 0
                    
                    # Bot损失（原有）
                    if cresci_mask.any():
                        task_loss += criterion_task(outputs['bot_logits'][cresci_mask], task_labels[cresci_mask])
                    
                    # 性别损失（加权）
                    if gender_mask.any():
                        gender_loss = criterion_task(outputs['gender_logits'][gender_mask], task_labels[gender_mask])
                        task_loss += gender_loss * gender_weight  # 加权提升性别分类训练优先级
                    
                    # 最终损失（降低域损失权重）
                    loss = lambda_task * task_loss + lambda_domain * domain_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM if hasattr(self.config, 'MAX_GRAD_NORM') else 1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    # 统计
                    train_loss += loss.item()
                    for i, domain in enumerate(domains):
                        pred = outputs['bot_logits'][i].argmax() if domain == 'cresci' else outputs['gender_logits'][i].argmax()
                        train_correct += (pred == task_labels[i]).item()
                    train_total += len(domains)
                    
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{train_correct/train_total:.4f}"}) if self.device.type != 'cpu' else None
                    
                    # 内存清理
                    if batch_idx % 50 == 0:
                        del input_ids, attention_mask, domain_labels, task_labels, outputs, loss, domain_loss, task_loss
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n⚠️  GPU内存不足! 清理后继续...")
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        gc.collect()
                        continue
                    else:
                        raise e
            
            # 记录历史
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_acc)
            
            print(f"\nEpoch {epoch+1} 训练结果 - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")
            
            # 评估当前模型
            current_eval_acc = self._eval_current_model()
            self._save_best_model(current_eval_acc, epoch)
            
            # 内存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # 训练完成后打印汇总
        print("\n✅ 训练完成!")
        print(f"📊 训练汇总:")
        print(f"   最优验证集准确率: {self.best_val_acc:.4f}")
        if len(self.history['domain_acc']) > 0:
            print(f"   最终域判别准确率: {self.history['domain_acc'][-1]:.4f}")
            print(f"   最终Bot检测准确率: {self.history['bot_acc'][-1]:.4f}")
            print(f"   最终性别分类准确率: {self.history['gender_acc'][-1]:.4f}")
        print(f"✓ 最优模型路径: {self.original_model_path}")

# ==================== 主函数 ====================
def main():
    from config import Config
    
    print("=" * 60)
    print("模型训练主程序 (支持从0训练/增量训练)")
    print("=" * 60)
    
    config = Config()
    trainer = DomainAdaptationTrainer(config)
    trainer.train()
    
    print("\n" + "=" * 60)
    print("✅ 训练流程完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()