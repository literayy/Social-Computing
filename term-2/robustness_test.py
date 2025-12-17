import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import random
from transformers import DebertaV2Tokenizer
# ==================== 解决中文乱码配置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统优先使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'


class RobustnessEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self._load_trained_model()
        self.tokenizer = self._load_tokenizer()
        
        self.test_df = pd.read_csv(self.config.PREPROCESSED_COMBINED)
        # 采样测试集
        self.test_df = self.test_df.sample(n=min(500, len(self.test_df)), random_state=42)
        
    def _load_tokenizer(self):
        """加载DeBERTa tokenizer"""
        from transformers import DebertaV2Tokenizer
        return DebertaV2Tokenizer.from_pretrained(self.config.DEBERTA_PATH)
    
    def _load_trained_model(self):
        """加载训练好的DomainAdaptiveDeBERTa模型（适配新的模型结构）"""
        # 导入训练定义的模型类
        from domain_adaptation import DomainAdaptiveDeBERTa  # 假设训练代码文件名为train_model.py
        
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, 'best_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 初始化模型
        model = DomainAdaptiveDeBERTa(self.config).to(self.device)
        
        # 加载状态字典（兼容不同的保存格式）
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✓ 加载模型: {model_path}")
        return model
    
    def adversarial_attack(self, text, attack_type='word_swap'):
        """
        生成文本对抗样本
        Args:
            text: 原始文本
            attack_type: 攻击类型 ('word_swap', 'char_insert')
        Returns:
            对抗样本文本
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return text
        
        if attack_type == 'word_swap':
            # 随机替换10%的词
            words = text.split()
            if len(words) <= 1:
                return text
            
            num_swaps = max(1, len(words) // 10)
            swap_indices = random.sample(range(len(words)), min(num_swaps, len(words)))
            
            common_words = ['good', 'bad', 'great', 'nice', 'love', 'hate', 'like', 'think', 'very', 'so']
            for idx in swap_indices:
                words[idx] = random.choice(common_words)
            
            return ' '.join(words)
        
        elif attack_type == 'char_insert':
            # 随机插入字符
            chars = list(text)
            if len(chars) <= 1:
                return text
            
            num_inserts = max(1, len(chars) // 20)
            for _ in range(num_inserts):
                pos = random.randint(0, len(chars)-1)
                chars.insert(pos, random.choice('abcdefghijklmnopqrstuvwxyz '))
            
            return ''.join(chars)
        
        return text
    
    def test_cross_domain_accuracy(self):
        """跨域泛化测试：Cresci测Bot，Gender测Gender"""
        print("\n=== 跨域泛化测试 ===")
        
        cresci_test = self.test_df[self.test_df['domain'] == 'cresci']
        gender_test = self.test_df[self.test_df['domain'] == 'gender']
        
        results = {}
        
        # Cresci → Cresci (同域Bot检测)
        if len(cresci_test) > 0:
            acc_cresci = self._evaluate_subset(cresci_test, task='bot')
            results['cresci_to_cresci'] = acc_cresci
            print(f"Cresci (同域) Bot检测准确率: {acc_cresci:.4f}")
        else:
            print("⚠️ Cresci域测试数据为空")
        
        # Gender → Gender (同域性别分类)
        if len(gender_test) > 0:
            acc_gender = self._evaluate_subset(gender_test, task='gender')
            results['gender_to_gender'] = acc_gender
            print(f"Gender (同域) 性别分类准确率: {acc_gender:.4f}")
        else:
            print("⚠️ Gender域测试数据为空")
        
        return results
    
    def test_adversarial_robustness(self):
        """对抗攻击鲁棒性测试"""
        print("\n=== 对抗攻击鲁棒性测试 ===")
        
        attack_types = ['word_swap', 'char_insert']
        results = {}
        
        # 原始准确率
        original_acc = self._evaluate_subset(self.test_df)
        results['original'] = original_acc
        print(f"原始准确率: {original_acc:.4f}")
        
        for attack_type in attack_types:
            print(f"\n攻击类型: {attack_type}")
            
            # 生成对抗样本
            attacked_df = self.test_df.copy()
            attacked_df['text'] = attacked_df['text'].apply(
                lambda x: self.adversarial_attack(x, attack_type)
            )
            
            # 评估对抗样本
            attacked_acc = self._evaluate_subset(attacked_df)
            results[attack_type] = attacked_acc
            
            drop_rate = (original_acc - attacked_acc) / original_acc if original_acc > 0 else 0
            print(f"攻击后准确率: {attacked_acc:.4f}")
            print(f"准确率下降: {drop_rate:.2%}")
        
        # 可视化结果
        self.plot_robustness_results(results)
        
        return results
    
    def _evaluate_subset(self, df, task='mixed'):
        """
        评估数据子集的准确率（适配新模型的前向推理）
        Args:
            df: 待评估的DataFrame
            task: 任务类型 ('mixed', 'bot', 'gender')
        Returns:
            准确率
        """
        if len(df) == 0:
            return 0.0
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="评估"):
                # 文本编码
                encoded = self.tokenizer(
                    row['text'],
                    padding='max_length',
                    truncation=True,
                    max_length=self.config.MAX_LENGTH,
                    return_tensors='pt'
                ).to(self.device)
                
                # 模型推理（适配新模型的forward参数）
                outputs = self.model(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    domain=[row['domain']]  # 保持与训练时一致的domain参数格式
                )
                
                # 根据域/任务选择分类器
                if row['domain'] == 'cresci' or task == 'bot':
                    pred = outputs['bot_logits'].argmax(dim=1).item()
                else:
                    pred = outputs['gender_logits'].argmax(dim=1).item()
                
                predictions.append(pred)
                true_labels.append(int(row['label']))
        
        accuracy = accuracy_score(true_labels, predictions)
        return accuracy
    
    def plot_robustness_results(self, results):
        """可视化鲁棒性测试结果"""
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 确保中文显示（根据系统调整）
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = ['original','word_swap','char_insert'] if len(results) == 3 else list(results.keys())
        values = list(results.values())
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        bars = ax.bar(labels, values, color=colors[:len(labels)], alpha=0.8)
        
        ax.set_ylabel('ACC', fontsize=12)  # 修正拼写错误 ACU → ACC
        ax.set_title('Robustness Test Results', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.OUTPUT_PATH, 'robustness_test.png')
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 鲁棒性测试图已保存: {save_path}")
    
    def generate_confusion_matrices(self):
        """生成并保存混淆矩阵（适配新模型）"""
        print("\n=== 生成混淆矩阵 ===")
        
        cresci_test = self.test_df[self.test_df['domain'] == 'cresci']
        gender_test = self.test_df[self.test_df['domain'] == 'gender']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bot检测混淆矩阵
        if len(cresci_test) > 0:
            predictions = []
            true_labels = []
            
            with torch.no_grad():
                for idx, row in cresci_test.iterrows():
                    encoded = self.tokenizer(
                        row['text'],
                        padding='max_length',
                        truncation=True,
                        max_length=self.config.MAX_LENGTH,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    outputs = self.model(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask'],
                        domain=['cresci']
                    )
                    
                    pred = outputs['bot_logits'].argmax(dim=1).item()
                    predictions.append(pred)
                    true_labels.append(int(row['label']))
            
            cm = confusion_matrix(true_labels, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                       xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])
            axes[0].set_title('Bot Detection Confusion Matrix', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Predicted Label')
            axes[0].set_ylabel('True Label')
        else:
            axes[0].text(0.5, 0.5, 'No Cresci Domain Data', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Bot Detection Confusion Matrix', fontsize=12, fontweight='bold')
        
        # 性别分类混淆矩阵
        if len(gender_test) > 0:
            predictions = []
            true_labels = []
            
            with torch.no_grad():
                for idx, row in gender_test.iterrows():
                    encoded = self.tokenizer(
                        row['text'],
                        padding='max_length',
                        truncation=True,
                        max_length=self.config.MAX_LENGTH,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    outputs = self.model(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask'],
                        domain=['gender']
                    )
                    
                    pred = outputs['gender_logits'].argmax(dim=1).item()
                    predictions.append(pred)
                    true_labels.append(int(row['label']))
            
            cm = confusion_matrix(true_labels, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                       xticklabels=['Male', 'Female', 'Brand'], 
                       yticklabels=['Male', 'Female', 'Brand'])
            axes[1].set_title('Gender Classification Confusion Matrix', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Predicted Label')
            axes[1].set_ylabel('True Label')
        else:
            axes[1].text(0.5, 0.5, 'No Gender Domain Data', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Gender Classification Confusion Matrix', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.config.OUTPUT_PATH, 'confusion_matrices.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 混淆矩阵已保存: {save_path}")
    
    def run(self):
        """执行完整的鲁棒性测试流程"""
        print("=" * 60)
        print("步骤5: 鲁棒性验证")
        print("=" * 60)
        
        # 执行各项测试
        cross_domain_results = self.test_cross_domain_accuracy()
        adversarial_results = self.test_adversarial_robustness()
        self.generate_confusion_matrices()
        
        print("\n" + "=" * 60)
        print("✅ 步骤5完成!")
        print("=" * 60)
        
        # 返回测试结果
        return {
            'cross_domain': cross_domain_results,
            'adversarial': adversarial_results
        }

# 主函数入口
if __name__ == "__main__":
    from config import Config
    # 初始化配置
    config = Config()
    
    # 创建评估器并运行测试
    evaluator = RobustnessEvaluator(config)
    test_results = evaluator.run()
    
    # 打印最终结果
    print("\n📊 测试结果汇总:")
    print(f"跨域测试结果: {test_results['cross_domain']}")
    print(f"对抗攻击测试结果: {test_results['adversarial']}")