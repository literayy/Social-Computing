import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from transformers import DebertaV2Tokenizer, DebertaV2Model
import warnings
import os
warnings.filterwarnings('ignore')


class GatedFusion(nn.Module):
    """门控融合模块 - 动态学习传统特征和DeBERTa特征的权重"""
    
    def __init__(self, trad_dim=50, bert_dim=1024, hidden_dim=512, dropout=0.1):
        super(GatedFusion, self).__init__()
        
        self.trad_dim = trad_dim
        self.bert_dim = bert_dim
        self.hidden_dim = hidden_dim
        
        # 投影到相同维度
        self.trad_proj = nn.Linear(trad_dim, hidden_dim)
        self.bert_proj = nn.Linear(bert_dim, hidden_dim)
        
        # 门控机制 - 学习两类特征的重要性权重
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 输出2个权重
            nn.Softmax(dim=-1)
        )
        
        # 特征交互层 - 学习特征间的交互
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, trad_features, bert_features):
        """
        Args:
            trad_features: (batch_size, trad_dim)
            bert_features: (batch_size, bert_dim)
        Returns:
            fused_features: (batch_size, hidden_dim)
        """
        # 投影到相同维度
        trad_proj = self.trad_proj(trad_features)  # (B, H)
        bert_proj = self.bert_proj(bert_features)  # (B, H)
        
        # 拼接用于门控和交互
        concat_features = torch.cat([trad_proj, bert_proj], dim=-1)  # (B, 2H)
        
        # 计算门控权重 - 动态决定两类特征的重要性
        gate_weights = self.gate(concat_features)  # (B, 2)
        
        # 加权融合
        weighted_trad = gate_weights[:, 0:1] * trad_proj  # (B, H)
        weighted_bert = gate_weights[:, 1:2] * bert_proj  # (B, H)
        
        # 特征交互 - 学习深层次的特征组合
        fused = self.interaction(concat_features)  # (B, H)
        
        # 残差连接 - 保留原始信息
        fused = fused + weighted_trad + weighted_bert
        
        # Dropout和Layer Normalization
        fused = self.dropout(fused)
        fused = self.layer_norm(fused)
        
        return fused, gate_weights  # 返回融合特征和门控权重（用于分析）


class FeatureExtractor:
    def __init__(self, config, use_attention_fusion=True, device='cuda'):
        self.config = config
        self.scaler = StandardScaler()
        self.use_attention_fusion = use_attention_fusion
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 初始化门控融合模块
        if use_attention_fusion:
            self.fusion_module = GatedFusion(
                trad_dim=50,
                bert_dim=1024,
                hidden_dim=512,
                dropout=0.1
            ).to(self.device)
            print(f"✓ 初始化门控融合模块 (Gated Fusion)")
            print(f"✓ 设备: {self.device}")
        
    def extract_traditional_features(self, df):
        """提取50维传统特征（原逻辑保持不变）"""
        print("\n提取传统特征...")
        features = pd.DataFrame()
        
        # 1. 账户基本特征 (10维)
        features['followers_count'] = df['followers_count'].fillna(0)
        features['friends_count'] = df['friends_count'].fillna(0)
        features['statuses_count'] = df['statuses_count'].fillna(0)
        features['favourites_count'] = df['favourites_count'].fillna(0)
        features['listed_count'] = df['listed_count'].fillna(0)
        
        # 粉丝/关注比例
        features['followers_friends_ratio'] = df.apply(
            lambda x: x['followers_count'] / (x['friends_count'] + 1), axis=1)
        
        # 平均每天发推数
        if 'created_at' in df.columns:
            df['account_age_days'] = df['created_at'].apply(self._calculate_account_age)
        else:
            df['account_age_days'] = 365  # 默认值
        
        features['tweets_per_day'] = df['statuses_count'] / (df['account_age_days'] + 1)
        features['account_age_days'] = df['account_age_days']
        
        # 是否认证
        features['verified'] = df['verified'].fillna(False).astype(int) if 'verified' in df.columns else 0
        features['default_profile'] = df['default_profile'].fillna(False).astype(int) if 'default_profile' in df.columns else 0
        
        # 2. 用户资料特征 (10维)
        features['name_length'] = df['name'].fillna('').astype(str).apply(len)
        features['screen_name_length'] = df['screen_name'].fillna('').astype(str).apply(len)
        features['description_length'] = df['description'].fillna('').astype(str).apply(len)
        
        # 用户名特征
        features['name_has_digits'] = df['name'].fillna('').astype(str).apply(
            lambda x: int(bool(re.search(r'\d', x))))
        features['screen_name_has_digits'] = df['screen_name'].fillna('').astype(str).apply(
            lambda x: int(bool(re.search(r'\d', x))))
        
        # 是否有描述
        features['has_description'] = (features['description_length'] > 0).astype(int)
        features['has_location'] = df['location'].fillna('').astype(str).apply(lambda x: int(len(x) > 0))
        features['has_url'] = df['url'].fillna('').astype(str).apply(lambda x: int(len(x) > 0)) if 'url' in df.columns else 0
        
        # 用户名和昵称相似度
        features['name_screen_name_similarity'] = df.apply(
            lambda x: self._string_similarity(str(x['name']), str(x['screen_name'])), axis=1)
        
        # 描述中的URL数量
        features['url_count_in_description'] = df['description'].fillna('').astype(str).apply(
            lambda x: len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x)))
        
        # 3. 推文内容特征 (15维)
        if 'all_tweets_text' in df.columns:
            features['total_tweet_length'] = df['all_tweets_text'].fillna('').astype(str).apply(len)
            features['avg_tweet_length'] = features['total_tweet_length'] / (features['statuses_count'] + 1)
            
            # 推文中的特殊字符
            features['hashtag_count'] = df['all_tweets_text'].fillna('').astype(str).apply(
                lambda x: len(re.findall(r'#\w+', x)))
            features['mention_count'] = df['all_tweets_text'].fillna('').astype(str).apply(
                lambda x: len(re.findall(r'@\w+', x)))
            features['url_count'] = df['all_tweets_text'].fillna('').astype(str).apply(
                lambda x: len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x)))
            
            # 每条推文的平均特殊字符数
            features['avg_hashtags_per_tweet'] = features['hashtag_count'] / (features['statuses_count'] + 1)
            features['avg_mentions_per_tweet'] = features['mention_count'] / (features['statuses_count'] + 1)
            features['avg_urls_per_tweet'] = features['url_count'] / (features['statuses_count'] + 1)
        else:
            for col in ['total_tweet_length', 'avg_tweet_length', 'hashtag_count', 
                       'mention_count', 'url_count', 'avg_hashtags_per_tweet',
                       'avg_mentions_per_tweet', 'avg_urls_per_tweet']:
                features[col] = 0
        
        # 聚合的推文统计特征
        if 'tweet_count' in df.columns:
            features['tweet_count'] = df['tweet_count'].fillna(0)
            features['avg_retweets'] = df['avg_retweets'].fillna(0)
            features['avg_favorites'] = df['avg_favorites'].fillna(0)
            features['max_retweets'] = df['max_retweets'].fillna(0)
            features['max_favorites'] = df['max_favorites'].fillna(0)
            features['total_retweets'] = df['total_retweets'].fillna(0)
            features['total_favorites'] = df['total_favorites'].fillna(0)
        else:
            for col in ['tweet_count', 'avg_retweets', 'avg_favorites', 
                       'max_retweets', 'max_favorites', 'total_retweets', 'total_favorites']:
                features[col] = 0
        
        # 4. 行为特征 (10维)
        features['retweet_ratio'] = features['total_retweets'] / (features['statuses_count'] + 1)
        features['favorite_ratio'] = features['total_favorites'] / (features['statuses_count'] + 1)
        
        # 互动率
        features['engagement_rate'] = (features['avg_retweets'] + features['avg_favorites']) / (features['followers_count'] + 1)
        
        # 活跃度
        features['following_rate'] = features['friends_count'] / (features['account_age_days'] + 1)
        features['listing_rate'] = features['listed_count'] / (features['followers_count'] + 1)
        # ========== 补充的5个行为特征==========
        # 1. 粉丝-关注差值率（反映账号“影响力倾向”：粉丝远多于关注为网红/营销号，反之可能是水军）
        features['followers_friends_diff_rate'] = (features['followers_count'] - features['friends_count']) / (features['followers_count'] + features['friends_count'] + 1)

        # 2. 推文互动效率（单条推文平均总互动数，反映内容质量）
        features['per_tweet_total_engagement'] = (features['total_retweets'] + features['total_favorites']) / (features['statuses_count'] + 1)

        # 3. 资料完整度（综合判断账号真实性：URL+位置+描述是否完善）
        features['profile_completeness'] = (
            features['has_url'] +  # 已有字段：是否有URL
            features['has_location'] +  # 已有字段：是否有位置
            (features['description_length'] > 0).astype(int)  # 描述是否非空
        ) / 3  # 归一化到0-1区间

        # 4. 高互动推文占比（反映账号是否有“爆款”内容能力）
        features['high_engagement_tweet_ratio'] = (features['max_retweets'] > 2 * features['avg_retweets']).astype(int)

        # 5. 收藏偏好度（反映账号是“内容生产者”还是“内容消费者”：收藏数/推文数越高，越偏向消费）
        features['favorites_per_tweet'] = features['favourites_count'] / (features['statuses_count'] + 1)
        # 5. 时序特征 (5维)
        if 'created_at' in df.columns:
            features['account_creation_year'] = df['created_at'].apply(self._extract_year)
            features['account_creation_month'] = df['created_at'].apply(self._extract_month)
            features['account_creation_hour'] = df['created_at'].apply(self._extract_hour)
        else:
            features['account_creation_year'] = 2015
            features['account_creation_month'] = 6
            features['account_creation_hour'] = 12
        
        # 账号是否新建
        features['is_new_account'] = (features['account_age_days'] < 30).astype(int)
        features['is_very_old_account'] = (features['account_age_days'] > 365*5).astype(int)
        
        print(f"提取了 {len(features.columns)} 维传统特征")
        
        # 处理无穷大和NaN值
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)
        
        return features
    
    def extract_deberta_features(self, df, text_column='all_tweets_text'):
        """加载并合并已有的DeBERTa分块特征"""
        print("\n加载并合并DeBERTa分块特征...")
        
        # 分块文件目录（与你保存的路径一致）
        deberta_chunks_dir = r"E:\social_calculate\output\deberta_chunks"
        
        # 检查目录是否存在
        if not os.path.exists(deberta_chunks_dir):
            raise FileNotFoundError(f"DeBERTa分块目录不存在: {deberta_chunks_dir}")
        
        # 获取所有分块文件并按序号排序
        chunk_files = sorted(
            [f for f in os.listdir(deberta_chunks_dir) if f.startswith('deberta_features_chunk_') and f.endswith('.npy')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])  # 按chunk序号排序
        )
        
        if not chunk_files:
            raise FileNotFoundError(f"未找到DeBERTa分块文件: {deberta_chunks_dir}")
        
        # 逐个加载并合并分块
        all_features = []
        for chunk_file in chunk_files:
            chunk_path = os.path.join(deberta_chunks_dir, chunk_file)
            chunk_features = np.load(chunk_path)
            all_features.append(chunk_features)
            print(f"  加载分块: {chunk_file}, 形状: {chunk_features.shape}")
        
        # 拼接为完整特征矩阵
        deberta_features = np.concatenate(all_features, axis=0)
        print(f"DeBERTa特征合并完成，总形状: {deberta_features.shape}")
        
        # 校验样本数与输入数据一致
        if len(deberta_features) != len(df):
            raise ValueError(
                f"DeBERTa特征样本数({len(deberta_features)})与输入数据样本数({len(df)})不匹配，请检查分块文件是否完整"
            )
        
        return deberta_features
    
    def combine_features(self, traditional_features, deberta_features):
        """
        使用门控融合机制合并传统特征和DeBERTa特征
        如果use_attention_fusion=False，则使用原始的拼接方式
        """
        if not self.use_attention_fusion:
            # 原始拼接方式（向后兼容）
            print("\n合并特征（简单拼接）...")
            traditional_scaled = self.scaler.fit_transform(traditional_features)
            combined = np.concatenate([traditional_scaled, deberta_features], axis=1)
            print(f"合并后特征维度: {combined.shape}")
            return combined, traditional_scaled
        
        # 使用门控融合
        print("\n使用门控融合机制合并特征...")
        
        # 标准化传统特征
        traditional_scaled = self.scaler.fit_transform(traditional_features)
        
        # 转换为Tensor
        trad_tensor = torch.FloatTensor(traditional_scaled).to(self.device)
        bert_tensor = torch.FloatTensor(deberta_features).to(self.device)
        
        # 批量处理（避免内存溢出）
        batch_size = 256
        fused_features_list = []
        gate_weights_list = []
        
        self.fusion_module.eval()
        with torch.no_grad():
            for i in range(0, len(trad_tensor), batch_size):
                batch_trad = trad_tensor[i:i+batch_size]
                batch_bert = bert_tensor[i:i+batch_size]
                
                # 门控融合
                batch_fused, batch_gates = self.fusion_module(batch_trad, batch_bert)
                fused_features_list.append(batch_fused.cpu().numpy())
                gate_weights_list.append(batch_gates.cpu().numpy())
                
                # 打印进度
                if (i // batch_size) % 10 == 0:
                    print(f"  处理进度: {i}/{len(trad_tensor)}")
        
        # 合并结果
        combined = np.concatenate(fused_features_list, axis=0)
        gate_weights = np.concatenate(gate_weights_list, axis=0)
        
        # 分析门控权重
        avg_trad_weight = gate_weights[:, 0].mean()
        avg_bert_weight = gate_weights[:, 1].mean()
        
        print(f"\n✓ 门控融合完成!")
        print(f"  原始维度: 50 (传统) + 1024 (DeBERTa) = 1074")
        print(f"  融合后维度: {combined.shape[1]}")
        print(f"  维度压缩率: {(1 - combined.shape[1]/1074)*100:.1f}%")
        print(f"\n  平均门控权重:")
        print(f"    传统特征: {avg_trad_weight:.3f} ({avg_trad_weight*100:.1f}%)")
        print(f"    DeBERTa特征: {avg_bert_weight:.3f} ({avg_bert_weight*100:.1f}%)")
        
        # 保存门控权重用于分析
        self.last_gate_weights = gate_weights
        
        return combined, traditional_scaled
    
    def save_fusion_module(self, save_path):
        """保存训练好的融合模块"""
        if self.use_attention_fusion:
            torch.save(self.fusion_module.state_dict(), save_path)
            print(f"✓ 融合模块已保存到: {save_path}")
    
    def load_fusion_module(self, load_path):
        """加载预训练的融合模块"""
        if self.use_attention_fusion:
            self.fusion_module.load_state_dict(torch.load(load_path, map_location=self.device))
            print(f"✓ 融合模块已加载: {load_path}")
    
    # 辅助函数（与原代码一致）
    def _calculate_account_age(self, created_at):
        try:
            if pd.isna(created_at) or created_at == '':
                return 365
            created = pd.to_datetime(created_at, errors='coerce')
            if pd.isna(created):
                return 365
            age = (datetime.now() - created).days
            return max(age, 1)
        except:
            return 365
    
    def _extract_year(self, created_at):
        try:
            return pd.to_datetime(created_at, errors='coerce').year
        except:
            return 2015
    
    def _extract_month(self, created_at):
        try:
            return pd.to_datetime(created_at, errors='coerce').month
        except:
            return 6
    
    def _extract_hour(self, created_at):
        try:
            return pd.to_datetime(created_at, errors='coerce').hour
        except:
            return 12
    
    def _string_similarity(self, s1, s2):
        """计算两个字符串的相似度"""
        s1, s2 = s1.lower(), s2.lower()
        if len(s1) == 0 or len(s2) == 0:
            return 0
        
        matches = sum(1 for a, b in zip(s1, s2) if a == b)
        return matches / max(len(s1), len(s2))


# 使用示例
if __name__ == "__main__":
    # 配置
    class Config:
        pass
    
    config = Config()
    
    # 方式1: 使用门控融合（推荐）
    extractor = FeatureExtractor(config, use_attention_fusion=True, device='cuda')
    
    # 方式2: 使用原始拼接（向后兼容）
    # extractor = FeatureExtractor(config, use_attention_fusion=False)
    
    # 假设你已经有预处理后的数据
    df = pd.read_csv("E:\social_calculate\output\preprocessed_data.csv")
    
    # 提取传统特征
    trad_features = extractor.extract_traditional_features(df)
    
    # 加载DeBERTa特征
    bert_features = extractor.extract_deberta_features(df)
    
    # 门控融合
    combined_features, _ = extractor.combine_features(trad_features, bert_features)
    # ========== 新增：保存融合特征到npy文件 ==========
    np.save("E:\social_calculate\output\combined_features.npy", combined_features)
    print(f"\n✓ 融合特征已保存为: combined_features.npy")
    print(f"  保存的特征形状: {combined_features.shape}")
    # 保存融合模块（可选）
    extractor.save_fusion_module("fusion_module.pth")