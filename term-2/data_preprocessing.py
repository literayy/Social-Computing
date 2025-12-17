import pandas as pd
import numpy as np
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        
    def load_cresci_data(self):
        """加载Cresci-2017数据集"""
        print("\n=== 加载Cresci-2017数据集 ===")
        
        # Cresci-2017数据集路径配置
        datasets = {
            'fake_followers': {'users': 'fake_followers.csv/fake_followers.csv/users.csv', 
                              'tweets': 'fake_followers.csv/fake_followers.csv/tweets.csv', 'label': 1},
            'genuine_accounts': {'users': 'genuine_accounts.csv/genuine_accounts.csv/users.csv',
                                'tweets': 'genuine_accounts.csv/genuine_accounts.csv/tweets.csv', 'label': 0},
            'social_spambots_1': {'users': 'social_spambots_1.csv/social_spambots_1.csv/users.csv',
                                 'tweets': 'social_spambots_1.csv/social_spambots_1.csv/tweets.csv', 'label': 1},
            'social_spambots_2': {'users': 'social_spambots_2.csv/social_spambots_2.csv/users.csv',
                                 'tweets': 'social_spambots_2.csv/social_spambots_2.csv/tweets.csv', 'label': 1},
            'social_spambots_3': {'users': 'social_spambots_3.csv/social_spambots_3.csv/users.csv',
                                 'tweets': 'social_spambots_3.csv/social_spambots_3.csv/tweets.csv', 'label': 1},
            'traditional_spambots_1': {'users': 'traditional_spambots_1.csv/traditional_spambots_1.csv/users.csv',
                                      'tweets': 'traditional_spambots_1.csv/traditional_spambots_1.csv/tweets.csv', 'label': 1},
        }
        
        all_data = []
        
        for dataset_name, paths in datasets.items():
            print(f"\n处理 {dataset_name}...")
            
            try:
                # 读取users
                user_path = os.path.join(self.config.CRESCI_PATH, 'datasets_full', paths['users'])
                users_df = pd.read_csv(user_path, encoding='utf-8', low_memory=False)
                
                # ===== 采样users，避免内存溢出 =====
                if len(users_df) > self.config.CRESCI_SAMPLE_SIZE:
                    users_df = users_df.sample(n=self.config.CRESCI_SAMPLE_SIZE, random_state=42)
                    print(f"  采样到 {self.config.CRESCI_SAMPLE_SIZE} 个用户")
                
                # 只保留需要的列
                keep_cols = ['id', 'name', 'description']
                users_df = users_df[[col for col in keep_cols if col in users_df.columns]]
                
                # 读取tweets（如果存在）
                if 'tweets' in paths:
                    tweet_path = os.path.join(self.config.CRESCI_PATH, 'datasets_full', paths['tweets'])
                    if os.path.exists(tweet_path):
                        # 只读取这些用户的tweets
                        user_ids = users_df['id'].tolist()
                        tweets_df = pd.read_csv(tweet_path, encoding='utf-8', low_memory=False)
                        tweets_df = tweets_df[tweets_df['user_id'].isin(user_ids)]
                        
                        # 每个用户只保留前3条tweets
                        tweets_df = tweets_df.groupby('user_id').head(3)
                        
                        # 只保留text列
                        tweets_df = tweets_df[['user_id', 'text']]
                        
                        # 聚合tweets
                        tweets_grouped = tweets_df.groupby('user_id')['text'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
                        tweets_grouped.columns = ['id', 'tweet_text']
                        
                        # 合并
                        merged_df = users_df.merge(tweets_grouped, on='id', how='left')
                    else:
                        merged_df = users_df
                        merged_df['tweet_text'] = ''
                else:
                    merged_df = users_df
                    merged_df['tweet_text'] = ''
                
                # 添加标签
                merged_df['label'] = paths['label']
                all_data.append(merged_df)
                
                print(f"✓ {dataset_name}: {len(merged_df)} 条")
                
            except Exception as e:
                print(f"✗ {dataset_name} 加载失败: {e}")
                import traceback
                traceback.print_exc()
        
        if not all_data:
            raise Exception("未能加载任何Cresci数据")
        
        cresci_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ Cresci总计: {len(cresci_df)} 条")
        print(f"Bot: {(cresci_df['label']==1).sum()}, Human: {(cresci_df['label']==0).sum()}")
        
        return cresci_df
    
    def load_gender_data(self):
        """加载Gender数据集"""
        print("\n=== 加载Gender数据集 ===")
        
        # 尝试多种编码方式
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk']
        df = None
        
        for encoding in encodings:
            try:
                print(f"尝试编码: {encoding}")
                df = pd.read_csv(self.config.GENDER_PATH, encoding=encoding)
                print(f"✓ 成功使用 {encoding} 编码")
                break
            except UnicodeDecodeError:
                print(f"✗ {encoding} 编码失败")
                continue
            except Exception as e:
                print(f"✗ {encoding} 编码出错: {e}")
                continue
        
        if df is None:
            raise Exception("无法使用任何已知编码读取Gender数据集")
        
        print(f"✓ Gender数据: {len(df)} 条")
        print(f"列名: {df.columns.tolist()}")
        
        return df
    
    def preprocess_cresci(self, df):
        """预处理Cresci数据"""
        print("\n=== 预处理Cresci数据 ===")
        
        # 合并所有文本字段
        text_parts = []
        
        if 'name' in df.columns:
            text_parts.append(df['name'].fillna(''))
        if 'description' in df.columns:
            text_parts.append(df['description'].fillna(''))
        if 'tweet_text' in df.columns:
            text_parts.append(df['tweet_text'].fillna(''))
        
        # 合并文本
        if text_parts:
            df['combined_text'] = text_parts[0]
            for part in text_parts[1:]:
                df['combined_text'] = df['combined_text'] + ' ' + part
        else:
            raise Exception("未找到可用的文本字段")
        
        # 清洗文本
        df['combined_text'] = df['combined_text'].apply(self._clean_text)
        
        # 过滤空文本
        df = df[df['combined_text'].str.len() > 10].copy()
        
        # 统一格式
        processed_df = pd.DataFrame({
            'user_id': df['id'].astype(str) if 'id' in df.columns else df.index.astype(str),
            'text': df['combined_text'],
            'label': df['label'],
            'domain': 'cresci'
        })
        
        print(f"✓ 预处理后: {len(processed_df)} 条")
        print(f"Bot: {(processed_df['label']==1).sum()}, Human: {(processed_df['label']==0).sum()}")
        
        return processed_df
    
    def preprocess_gender(self, df):
        """预处理Gender数据"""
        print("\n=== 预处理Gender数据 ===")
        
        # 合并text和description
        df['combined_text'] = (df['text'].fillna('') + ' ' + 
                              df['description'].fillna(''))
        
        # 清洗文本
        df['combined_text'] = df['combined_text'].apply(self._clean_text)
        
        # 过滤空文本和未知性别
        df = df[df['combined_text'].str.len() > 10].copy()
        df = df[df['gender'].isin(['male', 'female', 'brand'])].copy()
        
        # 性别映射为数值（用于后续多任务学习）
        gender_map = {'male': 0, 'female': 1, 'brand': 2}
        df['gender_label'] = df['gender'].map(gender_map)
        
        # 统一格式
        processed_df = pd.DataFrame({
            'user_id': df['_unit_id'].astype(str),
            'text': df['combined_text'],
            'label': df['gender_label'],  # 性别标签
            'domain': 'gender'
        })
        
        print(f"✓ 预处理后: {len(processed_df)} 条")
        print(f"Male: {(processed_df['label']==0).sum()}, "
              f"Female: {(processed_df['label']==1).sum()}, "
              f"Brand: {(processed_df['label']==2).sum()}")
        
        return processed_df
    
    def _clean_text(self, text):
        """文本清洗"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # 移除URL
        text = re.sub(r'http\S+|www\S+', '', text)
        # 移除@mentions
        text = re.sub(r'@\w+', '', text)
        # 移除特殊字符（保留基本标点）
        text = re.sub(r'[^\w\s,.!?]', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_data(self, cresci_df, gender_df):
        """数据探索性分析"""
        print("\n=== 数据分析 ===")
        
        # 文本长度统计
        cresci_df['text_length'] = cresci_df['text'].str.len()
        gender_df['text_length'] = gender_df['text'].str.len()
        
        print(f"\nCresci文本长度: 均值={cresci_df['text_length'].mean():.1f}, "
              f"中位数={cresci_df['text_length'].median():.1f}")
        print(f"Gender文本长度: 均值={gender_df['text_length'].mean():.1f}, "
              f"中位数={gender_df['text_length'].median():.1f}")
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 文本长度分布
        axes[0, 0].hist(cresci_df['text_length'], bins=50, alpha=0.5, label='Cresci', color='blue')
        axes[0, 0].hist(gender_df['text_length'], bins=50, alpha=0.5, label='Gender', color='orange')
        axes[0, 0].set_xlabel('Text Length')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Text Length Distribution')
        axes[0, 0].legend()
        
        # Cresci标签分布
        cresci_labels = cresci_df['label'].value_counts().sort_index()
        cresci_label_names = ['Human', 'Bot']
        axes[0, 1].bar(range(len(cresci_labels)), cresci_labels.values, 
                       tick_label=cresci_label_names, color='skyblue', alpha=0.7)
        axes[0, 1].set_xlabel('Label')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Cresci Label Distribution')
        for i, v in enumerate(cresci_labels.values):
            axes[0, 1].text(i, v + 50, str(v), ha='center', va='bottom')
        
        # Gender标签分布
        gender_labels = gender_df['label'].value_counts().sort_index()
        gender_label_names = ['Male', 'Female', 'Brand']
        axes[1, 0].bar(range(len(gender_labels)), gender_labels.values, 
                       tick_label=gender_label_names, color='lightcoral', alpha=0.7)
        axes[1, 0].set_xlabel('Label')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Gender Label Distribution')
        for i, v in enumerate(gender_labels.values):
            axes[1, 0].text(i, v + 20, str(v), ha='center', va='bottom')
        
        # 数据集大小对比
        dataset_sizes = [len(cresci_df), len(gender_df)]
        dataset_names = ['Cresci', 'Gender']
        axes[1, 1].bar(dataset_names, dataset_sizes, color=['skyblue', 'lightcoral'], alpha=0.7)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Dataset Size Comparison')
        for i, v in enumerate(dataset_sizes):
            axes[1, 1].text(i, v + 100, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.OUTPUT_PATH, 'data_analysis.png'), dpi=150)
        print(f"✓ 分析图保存至: {self.config.OUTPUT_PATH}/data_analysis.png")
        
    def merge_and_save(self, cresci_df, gender_df):
        """合并并保存数据"""
        print("\n=== 合并数据 ===")
        
        # 合并两个数据集
        combined_df = pd.concat([cresci_df, gender_df], ignore_index=True)
        
        # 随机打乱
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✓ 合并后总计: {len(combined_df)} 条")
        print(f"Cresci: {len(cresci_df)}, Gender: {len(gender_df)}")
        
        # 保存
        save_path = os.path.join(self.config.OUTPUT_PATH, 'preprocessed_combined_data.csv')
        combined_df.to_csv(save_path, index=False)
        print(f"✓ 保存至: {save_path}")
        
        # 分别保存（用于后续分域训练）
        cresci_path = os.path.join(self.config.OUTPUT_PATH, 'preprocessed_cresci.csv')
        gender_path = os.path.join(self.config.OUTPUT_PATH, 'preprocessed_gender.csv')
        
        cresci_df.to_csv(cresci_path, index=False)
        gender_df.to_csv(gender_path, index=False)
        
        print(f"✓ Cresci保存至: {cresci_path}")
        print(f"✓ Gender保存至: {gender_path}")
        
        return combined_df
    
    def run(self):
        """执行完整预处理流程"""
        print("=" * 60)
        print("步骤1: 双数据集预处理")
        print("=" * 60)
        
        # 加载数据
        cresci_raw = self.load_cresci_data()
        gender_raw = self.load_gender_data()
        
        # 预处理
        cresci_processed = self.preprocess_cresci(cresci_raw)
        gender_processed = self.preprocess_gender(gender_raw)
        
        # 分析
        self.analyze_data(cresci_processed, gender_processed)
        
        # 合并保存
        combined_df = self.merge_and_save(cresci_processed, gender_processed)
        
        print("\n" + "=" * 60)
        print("✅ 步骤1完成！")
        print("=" * 60)
        
        return combined_df


if __name__ == "__main__":
    from config import Config
    
    config = Config()
    preprocessor = DataPreprocessor(config)
    
    try:
        combined_data = preprocessor.run()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()