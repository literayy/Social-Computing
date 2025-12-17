"""
增强版用户画像生成模块 - 独立运行版本（内置聚类逻辑）
基于512维混合特征构建更精准的用户画像
流程：加载数据 → Bot过滤 → 自动寻优聚类 → 构建画像（无相似用户推荐）
"""
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class Config:
    """配置类"""
    def __init__(self):
        self.base_dir = r"E:\social_calculate"
        self.output_dir = os.path.join(self.base_dir, "output")
        self.model_dir = os.path.join(self.output_dir, "models")
        
        # 数据路径
        self.user_data_path = os.path.join(self.output_dir, "preprocessed_data.csv")
        self.fusion_features_path = os.path.join(self.output_dir, "combined_features.npy")
        
        # 模型路径
        self.xgboost_model_path = os.path.join(self.model_dir, "xgboost_model.pkl")
        
        # 聚类参数（自动寻优相关）
        self.pca_components = 50  # 聚类前特征降维维度（加速聚类）
        self.cluster_random_state = 42  # 聚类随机种子（保证结果可复现）
        self.k_search_range = range(4, 11)  # 聚类数寻优范围（4~10）
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)


class SimpleRecSystem:
    """简化的推荐系统类（只包含聚类结果，无推荐逻辑）"""
    def __init__(self, user_clusters):
        self.user_clusters = user_clusters


class EnhancedUserProfileBuilder:
    """
    增强版用户画像构建器（无相似用户推荐）
    核心特点：基于512维混合特征进行深度分析
    """
    
    def __init__(self, rec_system, user_df, fusion_features, traditional_features=None):
        """
        Args:
            rec_system: 推荐系统对象（仅包含聚类结果）
            user_df: 原始用户数据DataFrame
            fusion_features: 512维混合特征 (n_samples, 512)
            traditional_features: 50维传统特征 (可选，用于可解释性分析)
        """
        self.rec_system = rec_system
        self.user_df = user_df
        self.fusion_features = fusion_features
        self.traditional_features = traditional_features
        
        self.user_profiles = {}
        self.cluster_profiles = {}
        self.feature_importance = None
        
        print(f"✓ 初始化增强版用户画像构建器")
        print(f"  用户数: {len(user_df)}")
        print(f"  混合特征维度: {fusion_features.shape}")
    
    def build_all_user_profiles(self):
        """构建所有用户画像（无相似用户推荐）"""
        print("\n" + "="*60)
        print("构建基于混合特征的用户画像...")
        print("="*60)
        
        # 1. 分析特征重要性
        self._analyze_feature_importance()
        
        # 2. 构建簇级别画像
        self._build_enhanced_cluster_profiles()
        
        # 3. 构建个人画像
        for idx in range(len(self.user_df)):
            profile = self._build_enhanced_user_profile(idx)
            self.user_profiles[idx] = profile
            
            if (idx + 1) % 500 == 0:
                print(f"已构建 {idx + 1}/{len(self.user_df)} 个用户画像")
        
        print(f"✓ 完成所有用户画像构建")
        return self.user_profiles
    
    def _analyze_feature_importance(self):
        """
        分析混合特征的重要性
        使用方差分析找出最具区分度的特征维度
        """
        print("\n分析混合特征重要性...")
        
        # 计算每个特征维度的方差（方差大说明区分度高）
        feature_variance = np.var(self.fusion_features, axis=0)
        
        # 归一化到0-1
        self.feature_importance = feature_variance / np.max(feature_variance)
        
        # 找出最重要的维度
        top_k = 20
        top_indices = np.argsort(self.feature_importance)[-top_k:][::-1]
        
        print(f"✓ 前{top_k}个最重要的特征维度: {top_indices.tolist()}")
        print(f"  平均重要性: {self.feature_importance[top_indices].mean():.4f}")
    
    def _build_enhanced_cluster_profiles(self):
        """构建增强版簇画像"""
        print("\n构建增强版簇画像...")
        
        n_clusters = len(np.unique(self.rec_system.user_clusters))
        
        for cluster_id in range(n_clusters):
            cluster_mask = (self.rec_system.user_clusters == cluster_id)
            cluster_users = self.user_df[cluster_mask]
            cluster_features = self.fusion_features[cluster_mask]
            
            profile = {
                'cluster_id': cluster_id,
                'size': len(cluster_users),
                'percentage': len(cluster_users) / len(self.user_df) * 100,
            }
            
            # 1. 簇中心特征向量（512维）
            profile['centroid'] = np.mean(cluster_features, axis=0)
            
            # 2. 簇内聚度（簇内用户相似度）
            if len(cluster_features) > 1:
                # 计算簇内所有用户到中心的平均距离
                distances = cdist(cluster_features, [profile['centroid']], metric='euclidean')
                profile['cohesion'] = float(np.mean(distances))
                profile['std'] = float(np.std(distances))
            else:
                profile['cohesion'] = 0.0
                profile['std'] = 0.0
            
            # 3. 特征分布分析（找出该簇的特征特点）
            profile['distinctive_features'] = self._find_distinctive_features(
                cluster_features, cluster_id
            )
            
            # 4. 传统特征统计（用于可解释性）
            profile.update(self._extract_traditional_stats(cluster_users))
            
            # 5. 内容主题分析
            profile['topics'] = self._extract_cluster_topics(cluster_users)
            
            # 6. 簇标签（基于混合特征）
            profile['labels'] = self._generate_enhanced_cluster_labels(profile)
            
            self.cluster_profiles[cluster_id] = profile
            
            print(f"\n簇 {cluster_id} 画像:")
            print(f"  规模: {profile['size']}人 ({profile['percentage']:.1f}%)")
            print(f"  内聚度: {profile['cohesion']:.4f} (越小越紧密)")
            print(f"  影响力: {profile.get('influence_level', 'N/A')}")
            print(f"  标签: {', '.join(profile['labels'])}")
    
    def _find_distinctive_features(self, cluster_features, cluster_id):
        """
        找出该簇最具区分性的特征维度
        比较簇内特征均值与全局特征均值的差异
        """
        cluster_mean = np.mean(cluster_features, axis=0)
        global_mean = np.mean(self.fusion_features, axis=0)
        
        # 计算偏差（标准化）
        global_std = np.std(self.fusion_features, axis=0) + 1e-8
        deviation = (cluster_mean - global_mean) / global_std
        
        # 找出偏差最大的维度（正向和负向）
        top_positive = np.argsort(deviation)[-5:][::-1]  # 高于全局均值的特征
        top_negative = np.argsort(deviation)[:5]  # 低于全局均值的特征
        
        return {
            'high_features': top_positive.tolist(),
            'low_features': top_negative.tolist(),
            'high_values': deviation[top_positive].tolist(),
            'low_values': deviation[top_negative].tolist()
        }
    
    def _extract_traditional_stats(self, cluster_users):
        """提取传统统计特征（用于可解释性）"""
        stats = {}
        
        # 社交影响力
        if 'followers_count' in cluster_users.columns:
            stats['avg_followers'] = float(cluster_users['followers_count'].mean())
            stats['avg_friends'] = float(cluster_users['friends_count'].mean())
            stats['avg_statuses'] = float(cluster_users['statuses_count'].mean())
            
            if stats['avg_followers'] > 10000:
                stats['influence_level'] = '高影响力'
            elif stats['avg_followers'] > 1000:
                stats['influence_level'] = '中等影响力'
            else:
                stats['influence_level'] = '普通用户'
        
        # 账号年龄
        if 'account_age_days' in cluster_users.columns:
            stats['avg_account_age'] = float(cluster_users['account_age_days'].mean())
        
        # 认证比例
        if 'verified' in cluster_users.columns:
            stats['verified_ratio'] = float(cluster_users['verified'].mean())
        
        return stats
    
    def _extract_cluster_topics(self, cluster_users, top_n=5):
        """提取簇的主要话题"""
        if 'all_tweets_text' in cluster_users.columns:
            all_text = ' '.join(cluster_users['all_tweets_text'].fillna('').astype(str))
        elif 'description' in cluster_users.columns:
            all_text = ' '.join(cluster_users['description'].fillna('').astype(str))
        else:
            return ['未知主题']
        
        all_text = all_text.lower()
        all_text = re.sub(r'http\S+', '', all_text)
        all_text = re.sub(r'[^\w\s]', ' ', all_text)
        
        words = all_text.split()
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 
                      'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been',
                      'rt', 'http', 'https', 'com', 'www', 'just', 'get', 'like'}
        words = [w for w in words if len(w) > 3 and w not in stop_words]
        
        word_counts = Counter(words)
        topics = [word for word, count in word_counts.most_common(top_n)]
        
        return topics if topics else ['其他']
    
    def _generate_enhanced_cluster_labels(self, profile):
        """基于混合特征生成增强标签"""
        labels = []
        
        # 基于影响力
        if 'influence_level' in profile:
            labels.append(profile['influence_level'])
        
        # 基于内聚度（特征空间紧密度）
        if profile['cohesion'] < 5.0:
            labels.append('高度相似群体')
        elif profile['cohesion'] < 10.0:
            labels.append('中度相似群体')
        else:
            labels.append('松散群体')
        
        # 基于主题
        topics = profile.get('topics', [])
        topic_keywords = {
            '科技': ['tech', 'ai', 'software', 'code', 'data', 'app'],
            '新闻媒体': ['news', 'breaking', 'report', 'media'],
            '体育': ['sports', 'game', 'team', 'player'],
            '娱乐': ['music', 'movie', 'show', 'art'],
            '商业': ['business', 'market', 'startup'],
            '生活': ['life', 'food', 'travel', 'health']
        }
        
        for topic_name, keywords in topic_keywords.items():
            if any(kw in ' '.join(topics) for kw in keywords):
                labels.append(f'{topic_name}相关')
                break
        
        return labels if labels else ['普通用户']
    
    def _build_enhanced_user_profile(self, user_index):
        """构建增强版单个用户画像（无相似用户推荐）"""
        user_cluster = self.rec_system.user_clusters[user_index]
        cluster_profile = self.cluster_profiles[user_cluster]
        user_row = self.user_df.iloc[user_index]
        user_features = self.fusion_features[user_index]
        
        profile = {
            'user_index': user_index,
            'cluster': user_cluster,
            'cluster_labels': cluster_profile['labels'],
        }
        
        # 基础信息
        if 'id' in user_row:
            profile['user_id'] = str(user_row['id'])
        if 'screen_name' in user_row:
            profile['screen_name'] = str(user_row['screen_name'])
        
        # 1. 基于混合特征的相似度分析（仅用户与簇中心）
        centroid = cluster_profile['centroid']
        from sklearn.metrics.pairwise import cosine_similarity
        profile['similarity_to_center'] = float(
            cosine_similarity([user_features], [centroid])[0][0]
        )
        
        # 2. 特征突出性分析
        profile['distinctive_dims'] = self._analyze_user_distinctiveness(
            user_features, cluster_profile['centroid']
        )
        
        # 3. 传统特征（可解释性）
        profile.update(self._extract_user_traditional_features(user_row))
        
        # 4. 兴趣标签（继承簇主题）
        profile['interests'] = cluster_profile['topics'][:3]
        
        # 5. 综合标签
        profile['tags'] = self._generate_enhanced_user_tags(profile, cluster_profile)
        
        return profile
    
    def _analyze_user_distinctiveness(self, user_features, cluster_centroid):
        """分析用户在哪些特征维度上最突出"""
        # 计算与簇中心的偏差
        deviation = user_features - cluster_centroid
        
        # 找出偏差最大的维度
        abs_deviation = np.abs(deviation)
        top_dims = np.argsort(abs_deviation)[-5:][::-1]
        
        return {
            'dims': top_dims.tolist(),
            'values': deviation[top_dims].tolist()
        }
    
    def _extract_user_traditional_features(self, user_row):
        """提取用户的传统特征"""
        features = {}
        
        if 'followers_count' in user_row:
            features['followers'] = int(user_row['followers_count'])
            features['friends'] = int(user_row['friends_count'])
            features['statuses'] = int(user_row['statuses_count'])
            
            if features['followers'] > 10000:
                features['personal_influence'] = '高'
            elif features['followers'] > 1000:
                features['personal_influence'] = '中'
            else:
                features['personal_influence'] = '低'
        
        return features
    
    def _generate_enhanced_user_tags(self, user_profile, cluster_profile):
        """生成增强版用户标签"""
        tags = []
        
        # 簇标签
        tags.extend(cluster_profile['labels'][:2])
        
        # 基于相似度的标签
        similarity = user_profile['similarity_to_center']
        if similarity > 0.9:
            tags.append('典型代表')
        elif similarity < 0.7:
            tags.append('边缘用户')
        
        # 个性化标签
        if 'personal_influence' in user_profile:
            if user_profile['personal_influence'] == '高':
                tags.append('意见领袖')
        
        # 去重
        tags = list(dict.fromkeys(tags))
        
        return tags[:5]  # 最多5个标签
    
    def visualize_feature_space(self, save_path, method='tsne'):
        """
        可视化512维特征空间
        使用降维技术展示用户分布
        """
        print(f"\n使用{method.upper()}降维可视化特征空间...")
        
        # 降维
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            features_2d = reducer.fit_transform(self.fusion_features)
            explained_var = reducer.explained_variance_ratio_.sum()
            title = f'PCA可视化 (解释方差: {explained_var:.2%})'
        else:  # tsne
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = reducer.fit_transform(self.fusion_features)
            title = 't-SNE可视化'
        
        # 绘图
        plt.figure(figsize=(12, 10))
        
        scatter = plt.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=self.rec_system.user_clusters,
            cmap='tab10',
            alpha=0.6,
            s=30
        )
        
        plt.colorbar(scatter, label='簇ID')
        plt.xlabel('维度 1')
        plt.ylabel('维度 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # 标注簇中心
        for cluster_id in range(len(self.cluster_profiles)):
            cluster_mask = (self.rec_system.user_clusters == cluster_id)
            center = features_2d[cluster_mask].mean(axis=0)
            plt.scatter(center[0], center[1], 
                       marker='*', s=500, c='red', 
                       edgecolors='black', linewidths=2)
            plt.text(center[0], center[1], f'C{cluster_id}',
                    fontsize=12, fontweight='bold',
                    ha='center', va='center')
        
        plt.tight_layout()
        save_file = f"{save_path}/feature_space_{method}.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 特征空间可视化已保存: {save_file}")
    
    def visualize_cluster_profiles(self, save_path):
        """可视化簇画像（仅3张子图：规模、内聚度、平均粉丝数）"""
        print("\n生成簇画像可视化...")
        
        # 创建3张子图（2行2列，隐藏第4个）
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes[1, 1].axis('off')  # 隐藏第4个子图
        
        cluster_ids = list(self.cluster_profiles.keys())
        
        # 1. 簇大小分布
        ax = axes[0, 0]
        sizes = [self.cluster_profiles[cid]['size'] for cid in cluster_ids]
        ax.bar(cluster_ids, sizes, color='steelblue')
        ax.set_xlabel('簇ID')
        ax.set_ylabel('用户数')
        ax.set_title('各簇用户数量分布')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. 簇内聚度（基于512维特征）
        ax = axes[0, 1]
        cohesions = [self.cluster_profiles[cid]['cohesion'] for cid in cluster_ids]
        ax.bar(cluster_ids, cohesions, color='coral')
        ax.set_xlabel('簇ID')
        ax.set_ylabel('内聚度 (欧氏距离)')
        ax.set_title('各簇内聚度 (越小越紧密)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. 影响力分布（平均粉丝数）
        ax = axes[1, 0]
        if all('avg_followers' in self.cluster_profiles[cid] for cid in cluster_ids):
            followers = [self.cluster_profiles[cid]['avg_followers'] for cid in cluster_ids]
            ax.bar(cluster_ids, followers, color='lightgreen')
            ax.set_xlabel('簇ID')
            ax.set_ylabel('平均粉丝数')
            ax.set_title('各簇平均影响力')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_file = f"{save_path}/enhanced_cluster_profiles.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 簇画像可视化已保存: {save_file}")
    
    def generate_profile_report(self, save_path):
        """生成增强版用户画像报告（无相似用户推荐）"""
        report_path = f"{save_path}/enhanced_user_profile_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("增强版用户画像分析报告（基于512维混合特征）\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"总用户数: {len(self.user_df)}\n")
            f.write(f"簇数量: {len(self.cluster_profiles)}\n")
            f.write(f"特征维度: {self.fusion_features.shape[1]}\n\n")
            
            # 簇画像
            f.write("="*70 + "\n")
            f.write("各簇详细画像（基于混合特征分析）\n")
            f.write("="*70 + "\n\n")
            
            for cluster_id, profile in self.cluster_profiles.items():
                f.write(f"\n【簇 {cluster_id}】\n")
                f.write("-"*70 + "\n")
                f.write(f"规模: {profile['size']}人 ({profile['percentage']:.1f}%)\n")
                f.write(f"内聚度: {profile['cohesion']:.4f} (标准差: {profile['std']:.4f})\n")
                f.write(f"标签: {', '.join(profile['labels'])}\n")
                
                if 'influence_level' in profile:
                    f.write(f"影响力等级: {profile['influence_level']}\n")
                    f.write(f"平均粉丝数: {profile['avg_followers']:.0f}\n")
                
                f.write(f"主要话题: {', '.join(profile['topics'][:5])}\n")
                
                # 特征空间分析
                dist_features = profile['distinctive_features']
                f.write(f"特征突出维度: {dist_features['high_features'][:3]}\n")
            
            # 示例用户画像
            f.write("\n\n" + "="*70 + "\n")
            f.write("示例用户画像（前5个用户，基于混合特征）\n")
            f.write("="*70 + "\n\n")
            
            for i in range(min(5, len(self.user_profiles))):
                profile = self.user_profiles[i]
                f.write(f"\n用户 {i}:\n")
                f.write(f"  所属簇: {profile['cluster']}\n")
                f.write(f"  与簇中心相似度: {profile['similarity_to_center']:.4f}\n")
                f.write(f"  标签: {', '.join(profile['tags'])}\n")
                
                if 'personal_influence' in profile:
                    f.write(f"  影响力: {profile['personal_influence']}\n")
                
                f.write(f"  兴趣: {', '.join(profile['interests'])}\n")
        
        print(f"✓ 增强版用户画像报告已保存: {report_path}")
    
    def save_profiles(self, save_path):
        """保存用户画像数据（无相似用户信息）"""
        # 保存簇画像（移除numpy数组）
        cluster_file = f"{save_path}/enhanced_cluster_profiles.json"
        with open(cluster_file, 'w', encoding='utf-8') as f:
            profiles_dict = {}
            for cid, profile in self.cluster_profiles.items():
                # 过滤掉centroid（太大）
                filtered_profile = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else 
                        float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in profile.items()
                    if k != 'centroid'  # 不保存512维中心向量
                }
                profiles_dict[str(cid)] = filtered_profile
            json.dump(profiles_dict, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 增强版簇画像已保存: {cluster_file}")
        
        # 保存部分用户画像
        sample_profiles = {k: v for k, v in list(self.user_profiles.items())[:100]}
        user_file = f"{save_path}/enhanced_sample_user_profiles.json"
        
        with open(user_file, 'w', encoding='utf-8') as f:
            profiles_dict = {}
            for uid, profile in sample_profiles.items():
                # 移除相似用户相关（若有）
                filtered_profile = {k: v for k, v in profile.items() if k != 'similar_users'}
                profiles_dict[str(uid)] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else 
                        float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in filtered_profile.items()
                }
            json.dump(profiles_dict, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 增强版示例用户画像已保存: {user_file}")


def load_bot_detection_model(config):
    """加载Bot检测模型并进行预测"""
    print("\n" + "="*60)
    print("加载Bot检测模型...")
    print("="*60)
    
    model_path = config.xgboost_model_path
    if not os.path.exists(model_path):
        print(f"⚠️  警告：未找到模型文件 {model_path}")
        print("将对所有用户构建画像")
        return None
    
    # 加载模型
    print(f"加载模型: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ 模型加载成功: {type(model).__name__}")
    
    # 加载特征进行预测
    features_path = config.fusion_features_path
    if not os.path.exists(features_path):
        print(f"⚠️  警告：未找到特征文件 {features_path}")
        return None
    
    print(f"加载特征: {features_path}")
    features = np.load(features_path)
    print(f"✓ 特征形状: {features.shape}")
    
    # 预测
    print("进行Bot检测预测...")
    predictions = model.predict(features)
    
    # 统计结果
    n_total = len(predictions)
    n_human = (predictions == 0).sum()
    n_bot = (predictions == 1).sum()
    
    print(f"\n预测结果统计:")
    print(f"  总用户数: {n_total}")
    print(f"  真实用户(Human): {n_human} ({n_human/n_total*100:.1f}%)")
    print(f"  机器人(Bot): {n_bot} ({n_bot/n_total*100:.1f}%)")
    
    return predictions


def find_optimal_clusters(config, fusion_features):
    """
    自动寻找最优聚类数（基于轮廓系数）
    Args:
        config: 配置对象
        fusion_features: 过滤后的512维特征数组
    Returns:
        optimal_k: 最优聚类数
        cluster_labels: 最优聚类数对应的簇标签
    """
    print("\n" + "="*60)
    print("自动寻优最优聚类数...")
    print("="*60)
    
    # 先降维（加速寻优）
    if fusion_features.shape[1] > config.pca_components:
        print(f"对特征进行降维（{fusion_features.shape[1]}维 → {config.pca_components}维）...")
        pca = PCA(n_components=config.pca_components, random_state=config.cluster_random_state)
        features_reduced = pca.fit_transform(fusion_features)
        print(f"降维后特征方差解释率: {np.sum(pca.explained_variance_ratio_):.2%}")
    else:
        features_reduced = fusion_features
    
    # 遍历K值，计算轮廓系数
    silhouette_scores = {}
    inertia_scores = {}
    
    for k in config.k_search_range:
        print(f"测试聚类数 K={k}...")
        kmeans = KMeans(
            n_clusters=k,
            random_state=config.cluster_random_state,
            n_init=20,
            max_iter=500
        )
        cluster_labels = kmeans.fit_predict(features_reduced)
        
        # 计算轮廓系数（聚类质量指标，越接近1越好）
        if k > 1:
            silhouette = silhouette_score(features_reduced, cluster_labels)
            silhouette_scores[k] = silhouette
        inertia_scores[k] = kmeans.inertia_
        
        print(f"  K={k}: 轮廓系数={silhouette_scores.get(k, 0):.3f}, 内聚度={inertia_scores[k]:.2f}")
    
    # 选择轮廓系数最高的K作为最优值
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"\n✓ 最优聚类数: K={optimal_k} (轮廓系数={silhouette_scores[optimal_k]:.3f})")
    
    # 用最优K训练最终模型
    print(f"\n用最优K={optimal_k}训练最终聚类模型...")
    final_kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=config.cluster_random_state,
        n_init=20,
        max_iter=500
    )
    final_cluster_labels = final_kmeans.fit_predict(features_reduced)
    
    # 统计最终聚类结果
    cluster_counts = np.bincount(final_cluster_labels)
    print(f"✓ 最终聚类结果分布:")
    for i, count in enumerate(cluster_counts):
        print(f"  簇{i}: {count}人 ({count/len(final_cluster_labels)*100:.1f}%)")
    
    # 保存聚类结果
    cluster_save_path = os.path.join(config.output_dir, "user_clusters_generated.npy")
    np.save(cluster_save_path, final_cluster_labels)
    print(f"✓ 最优聚类结果已保存至: {cluster_save_path}")
    
    return optimal_k, final_cluster_labels


def main():
    """主程序入口"""
    print("="*70)
    print("增强版用户画像构建系统 - 独立运行版本")
    print("基于512维混合特征 + Bot检测 + 自动寻优聚类（无相似推荐）")
    print("流程：加载数据 → Bot过滤 → 自动寻优聚类 → 构建画像")
    print("="*70)
    
    # 1. 初始化配置
    config = Config()
    print(f"\n配置信息:")
    print(f"  基础目录: {config.base_dir}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  模型目录: {config.model_dir}")
    print(f"  聚类寻优范围: K={config.k_search_range.start}~{config.k_search_range.stop-1}")
    
    # 2. 检查核心文件
    print("\n" + "="*60)
    print("检查核心文件...")
    print("="*60)
    
    required_files = {
        '用户数据': config.user_data_path,
        '混合特征': config.fusion_features_path,
        'XGBoost模型': config.xgboost_model_path  # Bot检测模型可选
    }
    
    missing_files = []
    for name, path in required_files.items():
        if name == 'XGBoost模型' and not os.path.exists(path):
            print(f"⚠️ {name}: {path} [缺失，将跳过Bot检测]")
            continue
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} [缺失]")
            missing_files.append(name)
    
    if missing_files:
        print(f"\n❌ 错误：缺失以下核心文件: {', '.join(missing_files)}")
        print("请确保以下文件存在:")
        for name in missing_files:
            print(f"  - {required_files[name]}")
        return
    
    # 3. 加载数据
    print("\n" + "="*60)
    print("加载数据...")
    print("="*60)
    
    # 加载用户数据
    print(f"加载用户数据: {config.user_data_path}")
    user_df = pd.read_csv(config.user_data_path)
    print(f"✓ 用户数据形状: {user_df.shape}")
    print(f"  列名: {list(user_df.columns[:10])}...")
    
    # 加载混合特征
    print(f"\n加载混合特征: {config.fusion_features_path}")
    fusion_features = np.load(config.fusion_features_path)
    print(f"✓ 混合特征形状: {fusion_features.shape}")
    
    # 4. Bot检测（可选）
    predictions = load_bot_detection_model(config)
    
    # 5. 过滤真实用户
    if predictions is not None:
        human_mask = (predictions == 0)  # 0表示Human
        print(f"\n⚠️  过滤Bot用户，仅保留真实用户")
        
        # 过滤数据
        user_df_filtered = user_df[human_mask].reset_index(drop=True)
        fusion_features_filtered = fusion_features[human_mask]
        
        print(f"✓ 过滤后用户数: {len(user_df_filtered)}")
        print(f"✓ 过滤后特征形状: {fusion_features_filtered.shape}")
    else:
        print("\n⚠️  未进行Bot检测，将对所有用户进行聚类和画像")
        user_df_filtered = user_df
        fusion_features_filtered = fusion_features
    
    # 6. 自动寻优聚类（核心调整：替代固定K=8）
    optimal_k, cluster_labels_filtered = find_optimal_clusters(config, fusion_features_filtered)
    
    # 7. 创建简化的推荐系统对象（仅聚类结果）
    rec_system = SimpleRecSystem(cluster_labels_filtered)
    
    # 8. 构建用户画像
    print("\n" + "="*60)
    print("开始构建用户画像...")
    print("="*60)
    
    profile_builder = EnhancedUserProfileBuilder(
        rec_system=rec_system,
        user_df=user_df_filtered,
        fusion_features=fusion_features_filtered
    )
    
    # 9. 构建所有用户画像
    user_profiles = profile_builder.build_all_user_profiles()
    
    # 10. 生成可视化
    print("\n" + "="*60)
    print("生成可视化结果...")
    print("="*60)
    
    # t-SNE可视化
    profile_builder.visualize_feature_space(
        save_path=config.output_dir,
        method='tsne'
    )
    
    # PCA可视化
    profile_builder.visualize_feature_space(
        save_path=config.output_dir,
        method='pca'
    )
    
    # 簇画像可视化（仅3张子图）
    profile_builder.visualize_cluster_profiles(
        save_path=config.output_dir
    )
    
    # 11. 生成报告
    profile_builder.generate_profile_report(
        save_path=config.output_dir
    )
    
    # 12. 保存画像数据
    profile_builder.save_profiles(
        save_path=config.output_dir
    )
    
    # 13. 输出统计摘要
    print("\n" + "="*60)
    print("用户画像构建完成摘要")
    print("="*60)
    print(f"✓ 最终分析用户数: {len(user_df_filtered)}")
    print(f"✓ 生成用户画像数: {len(user_profiles)}")
    print(f"✓ 生成簇画像数: {len(profile_builder.cluster_profiles)}")
    print(f"✓ 特征维度: {fusion_features_filtered.shape[1]}")
    print(f"✓ 最优聚类簇数量: {optimal_k}")
    print(f"✓ 结果保存至: {config.output_dir}")
    
    # 14. 展示示例画像（无相似用户）
    print("\n" + "="*60)
    print("示例用户画像预览")
    print("="*60)
    
    for i in range(min(3, len(user_profiles))):
        profile = user_profiles[i]
        print(f"\n【用户 {i}】")
        print(f"  所属簇: {profile['cluster']}")
        print(f"  标签: {', '.join(profile['tags'])}")
        print(f"  与簇中心相似度: {profile['similarity_to_center']:.4f}")
        
        if 'personal_influence' in profile:
            print(f"  影响力: {profile['personal_influence']}")
        
        print(f"  兴趣: {', '.join(profile['interests'])}")
    
    print("\n" + "="*60)
    print("✓ 增强版用户画像构建完成！")
    print("="*60)
    print(f"\n生成的文件:")
    print(f"  1. {config.output_dir}/feature_space_tsne.png")
    print(f"  2. {config.output_dir}/feature_space_pca.png")
    print(f"  3. {config.output_dir}/enhanced_cluster_profiles.png")
    print(f"  4. {config.output_dir}/enhanced_user_profile_report.txt")
    print(f"  5. {config.output_dir}/enhanced_cluster_profiles.json")
    print(f"  6. {config.output_dir}/enhanced_sample_user_profiles.json")
    print(f"  7. {config.output_dir}/user_clusters_generated.npy (最优聚类结果)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n请检查:")
        print("  1. 核心数据文件是否存在")
        print("  2. 数据维度是否匹配（用户数与特征数）")
        print("  3. 聚类寻优范围是否合理")