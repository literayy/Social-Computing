import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        
    def load_all_datasets(self):
        """加载所有数据集"""
        all_users = []
        all_tweets = []
        
        print("开始加载数据集...")
        for dataset_name, paths in self.config.DATASETS.items():
            print(f"加载 {dataset_name}...")
            
            try:
                # 加载用户数据
                if paths['users'] and pd.io.common.file_exists(paths['users']):
                    users_df = pd.read_csv(paths['users'], encoding='utf-8', low_memory=False)
                    users_df['dataset'] = dataset_name
                    users_df['label'] = 1 if paths['label'] == 'fake' else 0
                    all_users.append(users_df)
                    print(f"  用户数据: {len(users_df)} 条")
                
                # 加载推文数据
                if paths['tweets'] and pd.io.common.file_exists(paths['tweets']):
                    tweets_df = pd.read_csv(paths['tweets'], encoding='utf-8', low_memory=False)
                    tweets_df['dataset'] = dataset_name
                    all_tweets.append(tweets_df)
                    print(f"  推文数据: {len(tweets_df)} 条")
            except Exception as e:
                print(f"  加载 {dataset_name} 时出错: {str(e)}")
                continue
        
        # 合并所有数据
        users_data = pd.concat(all_users, ignore_index=True) if all_users else pd.DataFrame()
        tweets_data = pd.concat(all_tweets, ignore_index=True) if all_tweets else pd.DataFrame()
        
        print(f"\n总用户数: {len(users_data)}")
        print(f"总推文数: {len(tweets_data)}")
        print(f"标签分布:\n{users_data['label'].value_counts()}")
        
        return users_data, tweets_data
    
    def clean_data(self, users_df, tweets_df):
        """数据清洗"""
        print("\n开始数据清洗...")
        
        # 用户数据清洗
        if not users_df.empty:
            # 删除重复用户
            users_df = users_df.drop_duplicates(subset=['id'], keep='first')
            
            # 处理缺失值
            numeric_cols = users_df.select_dtypes(include=[np.number]).columns
            users_df[numeric_cols] = users_df[numeric_cols].fillna(0)
            
            # 处理文本字段
            text_cols = ['name', 'screen_name', 'description', 'location']
            for col in text_cols:
                if col in users_df.columns:
                    users_df[col] = users_df[col].fillna('').astype(str)
            
            print(f"清洗后用户数: {len(users_df)}")
        
        # 推文数据清洗
        if not tweets_df.empty:
            # 删除重复推文
            tweets_df = tweets_df.drop_duplicates(subset=['id'], keep='first')
            
            # 处理文本字段
            if 'text' in tweets_df.columns:
                tweets_df['text'] = tweets_df['text'].fillna('').astype(str)
            
            print(f"清洗后推文数: {len(tweets_df)}")
        
        return users_df, tweets_df
    
    def aggregate_tweet_features(self, users_df, tweets_df):
        """聚合每个用户的推文特征"""
        print("\n聚合推文特征...")
        
        if tweets_df.empty or 'user_id' not in tweets_df.columns:
            print("推文数据为空或缺少user_id字段")
            return users_df
        
        # 按用户聚合推文统计
        tweet_stats = tweets_df.groupby('user_id').agg({
            'id': 'count',  # 推文数量
            'text': lambda x: ' '.join(x.astype(str)),  # 合并所有推文文本
            'retweet_count': ['mean', 'sum', 'max'] if 'retweet_count' in tweets_df.columns else 'count',
            'favorite_count': ['mean', 'sum', 'max'] if 'favorite_count' in tweets_df.columns else 'count',
        }).reset_index()
        
        # 重命名列
        tweet_stats.columns = ['user_id', 'tweet_count', 'all_tweets_text', 
                               'avg_retweets', 'total_retweets', 'max_retweets',
                               'avg_favorites', 'total_favorites', 'max_favorites']
        
        # 合并到用户数据
        users_df = users_df.merge(tweet_stats, left_on='id', right_on='user_id', how='left')
        
        # 填充缺失值
        numeric_cols = ['tweet_count', 'avg_retweets', 'total_retweets', 'max_retweets',
                       'avg_favorites', 'total_favorites', 'max_favorites']
        for col in numeric_cols:
            if col in users_df.columns:
                users_df[col] = users_df[col].fillna(0)
        
        if 'all_tweets_text' in users_df.columns:
            users_df['all_tweets_text'] = users_df['all_tweets_text'].fillna('')
        
        print(f"聚合完成，新增特征数: {len(numeric_cols)}")
        
        return users_df
    
    def preprocess(self):
        """完整的预处理流程"""
        # 加载数据
        users_df, tweets_df = self.load_all_datasets()
        
        # 清洗数据
        users_df, tweets_df = self.clean_data(users_df, tweets_df)
        
        # 聚合推文特征
        users_df = self.aggregate_tweet_features(users_df, tweets_df)
        
        # 保存预处理后的数据
        output_file = f"{self.config.OUTPUT_PATH}/preprocessed_data.csv"
        users_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n预处理完成! 数据已保存到: {output_file}")
        
        return users_df