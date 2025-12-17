"""
真实用户个性化推荐系统 + 效果验证 - 独立运行版本
核心特点：内置Bot过滤 | 自动生成标签 | 路径与用户画像模块一致
（移除推荐多样性验证）
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    pairwise_distances, adjusted_rand_score
)
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置类（与用户画像模块路径一致） ====================
class Config:
    """配置类 - 路径与用户画像模块保持统一"""
    def __init__(self):
        self.base_dir = r"E:\social_calculate"
        self.output_dir = os.path.join(self.base_dir, "output")
        self.model_dir = os.path.join(self.output_dir, "models")
        
        # 数据路径
        self.user_data_path = os.path.join(self.output_dir, "preprocessed_data.csv")
        self.fusion_features_path = os.path.join(self.output_dir, "combined_features.npy")
        self.xgboost_model_path = os.path.join(self.model_dir, "xgboost_model.pkl")
        
        # 结果路径
        self.RESULT_PATH = self.output_dir
        self.MODEL_PATH = self.model_dir
        
        # 聚类参数
        self.max_k = 10
        self.random_state = 42
        
        # 确保目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

# ==================== 推荐系统核心类 ====================
class UserRecommendationSystem:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.kmeans = None
        self.knn_model = None
        self.user_clusters = None
        self.cluster_profiles = {}
        self.validation_results = {}
        self.genuine_features = None
        self.genuine_df = None

    # ==================== 新增：Bot检测过滤 ====================
    def load_bot_detection_model(self):
        """加载Bot检测模型并预测"""
        print("\n" + "="*60)
        print("第一步：加载Bot检测模型过滤真实用户")
        print("="*60)
        
        if not os.path.exists(self.config.xgboost_model_path):
            raise FileNotFoundError(f"Bot检测模型缺失: {self.config.xgboost_model_path}")
        
        # 加载模型
        with open(self.config.xgboost_model_path, 'rb') as f:
            model = pickle.load(f)
        
        # 加载特征
        features = np.load(self.config.fusion_features_path)
        print(f"✓ 加载混合特征: {features.shape}")
        
        # 预测Bot/真实用户
        predictions = model.predict(features)
        self.genuine_mask = (predictions == 0)  # 0=真实用户
        
        # 统计结果
        n_total = len(predictions)
        n_genuine = self.genuine_mask.sum()
        n_bot = n_total - n_genuine
        
        print(f"✓ Bot检测完成:")
        print(f"  总用户数: {n_total}")
        print(f"  真实用户数: {n_genuine} ({n_genuine/n_total*100:.1f}%)")
        print(f"  Bot用户数: {n_bot} ({n_bot/n_total*100:.1f}%)")
        
        return predictions

    # ==================== 数据加载（基于Bot过滤结果） ====================
    def load_genuine_users(self):
        """加载过滤后的真实用户数据"""
        # 加载原始数据
        features = np.load(self.config.fusion_features_path)
        df = pd.read_csv(self.config.user_data_path)
        
        # 应用Bot过滤掩码
        self.genuine_features = features[self.genuine_mask]
        self.genuine_df = df[self.genuine_mask].reset_index(drop=True)
        
        # 特征标准化
        self.genuine_features = self.scaler.fit_transform(self.genuine_features)
        # L2归一化（提升余弦相似度效果）
        self.genuine_features = normalize(self.genuine_features, norm='l2')
        
        print(f"\n✓ 加载真实用户数据完成:")
        print(f"  特征维度: {self.genuine_features.shape}")
        print(f"  用户数据列: {list(self.genuine_df.columns[:10])}...")
        
        return self.genuine_features, self.genuine_df

    # ==================== 原有方法（保持不变） ====================
    def find_optimal_clusters(self, features, max_k=10):
        """寻找最优聚类数"""
        print("\n寻找最优聚类数...")
        
        inertias = []
        silhouette_scores = []
        k_range = range(4, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(features, kmeans.labels_))
            print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
        
        # 绘制评估图
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_xlabel('聚类数 K')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('肘部法则')
        axes[0].grid(True)
        
        axes[1].plot(k_range, silhouette_scores, 'ro-')
        axes[1].set_xlabel('聚类数 K')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('轮廓系数')
        axes[1].grid(True)
        
        plt.tight_layout()
        save_path = f"{self.config.RESULT_PATH}/cluster_evaluation.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 聚类评估图已保存: {save_path}")
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\n推荐聚类数: K={optimal_k}")
        
        return optimal_k

    def cluster_users(self, n_clusters=None):
        """用户聚类 + 自动生成标签文件"""
        print("\n" + "="*60)
        print("用户聚类分析...")
        print("="*60)
        
        features = self.genuine_features
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(features, max_k=self.config.max_k)
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_state, n_init=10)
        self.user_clusters = self.kmeans.fit_predict(features)
        
        # 自动保存聚类标签文件
        cluster_label_path = os.path.join(self.config.RESULT_PATH, "recommendation_clusters.npy")
        np.save(cluster_label_path, self.user_clusters)
        print(f"✓ 聚类标签已自动保存: {cluster_label_path}")
        
        # 聚类指标计算
        silhouette = silhouette_score(features, self.user_clusters)
        davies_bouldin = davies_bouldin_score(features, self.user_clusters)
        calinski = calinski_harabasz_score(features, self.user_clusters)
        
        print(f"\n聚类结果:")
        print(f"  聚类数: {n_clusters}")
        print(f"  轮廓系数: {silhouette:.3f} (越高越好)")
        print(f"  Davies-Bouldin指数: {davies_bouldin:.3f} (越低越好)")
        print(f"  Calinski-Harabasz分数: {calinski:.2f} (越高越好)")
        
        return self.user_clusters

    def visualize_clusters(self):
        """可视化聚类结果"""
        print("\n可视化聚类结果...")
        features = self.genuine_features
        
        pca = PCA(n_components=2, random_state=self.config.random_state)
        features_2d = pca.fit_transform(features)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            features_2d[:, 0], 
            features_2d[:, 1], 
            c=self.user_clusters, 
            cmap='viridis', 
            alpha=0.6,
            s=50
        )
        plt.colorbar(scatter, label='Cluster ID')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title('真实用户聚类可视化 (PCA降维)')
        plt.grid(True, alpha=0.3)
        
        # 标注簇中心
        centers_2d = pca.transform(self.kmeans.cluster_centers_)
        for i, center in enumerate(centers_2d):
            plt.scatter(center[0], center[1], c='red', marker='*', s=200, edgecolors='black')
            plt.text(center[0], center[1], f'C{i}', fontsize=12, fontweight='bold')
        
        save_path = f"{self.config.RESULT_PATH}/user_clusters_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 聚类可视化图已保存: {save_path}")

    def analyze_cluster_profiles(self):
        """分析簇画像"""
        print("\n" + "="*60)
        print("分析用户簇画像...")
        print("="*60)
        
        features = self.genuine_features
        df = self.genuine_df
        n_clusters = len(np.unique(self.user_clusters))
        
        for cluster_id in range(n_clusters):
            cluster_mask = (self.user_clusters == cluster_id)
            cluster_users = df[cluster_mask]
            cluster_size = len(cluster_users)
            
            # 计算簇中心和内聚度
            cluster_center = self.kmeans.cluster_centers_[cluster_id]
            cluster_features = features[cluster_mask]
            distances = pairwise_distances(cluster_features, [cluster_center], metric='cosine')
            cohesion = np.mean(distances)
            
            # 统计传统特征
            profile = {
                'cluster_id': cluster_id,
                'size': cluster_size,
                'percentage': cluster_size / len(df) * 100,
                'cohesion': cohesion,
                'centroid': cluster_center.tolist()
            }
            
            if 'followers_count' in cluster_users.columns:
                profile['avg_followers'] = cluster_users['followers_count'].mean()
                profile['avg_friends'] = cluster_users['friends_count'].mean()
                profile['avg_statuses'] = cluster_users['statuses_count'].mean()
            
            if 'tweets_per_day' in cluster_users.columns:
                profile['avg_tweets'] = cluster_users['tweets_per_day'].mean()
            
            self.cluster_profiles[cluster_id] = profile
            
            print(f"\n簇 {cluster_id}:")
            print(f"  规模: {cluster_size}人 ({profile['percentage']:.1f}%)")
            print(f"  内聚度: {cohesion:.3f} (越低越紧密)")
            if 'avg_followers' in profile:
                print(f"  平均粉丝数: {profile['avg_followers']:.0f}")
            if 'avg_tweets' in profile:
                print(f"  日均推文数: {profile['avg_tweets']:.2f}")

    def build_knn_recommender(self, n_neighbors=10):
        """构建KNN推荐模型（余弦相似度）"""
        print("\n构建KNN推荐模型...")
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.knn_model.fit(self.genuine_features)
        print(f"✓ KNN模型构建完成 (K={n_neighbors}, 相似度度量: 余弦)")
        return self.knn_model

    def recommend_similar_users(self, user_index, n_recommendations=5):
        """推荐相似用户"""
        if self.knn_model is None:
            raise ValueError("请先调用build_knn_recommender构建模型！")
        
        distances, indices = self.knn_model.kneighbors(
            self.genuine_features[user_index].reshape(1, -1),
            n_neighbors=n_recommendations + 1
        )
        
        similar_indices = indices[0][1:]  # 排除自身
        similar_distances = distances[0][1:]
        
        recommendations = []
        for idx, dist in zip(similar_indices, similar_distances):
            user_info = {
                'user_index': int(idx),
                'similarity': float(1 - dist),  # 余弦距离转相似度
                'cluster': int(self.user_clusters[idx])
            }
            
            if 'id' in self.genuine_df.columns:
                user_info['user_id'] = str(self.genuine_df.iloc[idx]['id'])
            if 'screen_name' in self.genuine_df.columns:
                user_info['screen_name'] = str(self.genuine_df.iloc[idx]['screen_name'])
            
            recommendations.append(user_info)
        
        return recommendations

    # ==================== 验证方法（移除多样性验证） ====================
    def validate_clustering_quality(self):
        """验证1: 聚类质量"""
        print("\n" + "="*60)
        print("验证1: 聚类质量评估")
        print("="*60)
        
        features = self.genuine_features
        clusters = self.user_clusters
        
        silhouette = silhouette_score(features, clusters)
        davies_bouldin = davies_bouldin_score(features, clusters)
        calinski = calinski_harabasz_score(features, clusters)
        
        unique, counts = np.unique(clusters, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        size_std = np.std(counts)
        size_balance_score = 1 - (size_std / np.mean(counts)) if np.mean(counts) != 0 else 0
        
        results = {
            'silhouette_score': silhouette,
            'davies_bouldin_index': davies_bouldin,
            'calinski_harabasz_score': calinski,
            'cluster_sizes': cluster_sizes,
            'size_balance_score': size_balance_score
        }
        
        print(f"\n聚类质量指标:")
        print(f"  轮廓系数: {silhouette:.4f} ([-1,1], 越高越好)")
        print(f"  Davies-Bouldin指数: {davies_bouldin:.4f} (越小越好)")
        print(f"  Calinski-Harabasz分数: {calinski:.2f} (越高越好)")
        print(f"  簇大小均衡度: {size_balance_score:.4f} ([0,1], 越高越均衡)")
        
        # 评分
        score = 0
        if silhouette > 0.3:
            score += 25
        elif silhouette > 0.2:
            score += 15
        
        if davies_bouldin < 1.0:
            score += 25
        elif davies_bouldin < 1.5:
            score += 15
        
        if size_balance_score > 0.7:
            score += 25
        elif size_balance_score > 0.5:
            score += 15
        
        results['clustering_score'] = score
        print(f"\n聚类质量总分: {score}/75")
        
        self.validation_results['clustering_quality'] = results
        return results

    def validate_recommendation_similarity(self, n_samples=100, n_recommendations=5):
        """验证2: 推荐相似度"""
        print("\n" + "="*60)
        print("验证2: 推荐相似度验证")
        print("="*60)
        
        features = self.genuine_features
        df = self.genuine_df
        sample_indices = np.random.choice(
            len(features), 
            size=min(n_samples, len(features)), 
            replace=False
        )
        
        intra_cluster_ratios = []
        similarity_scores = []
        
        for idx in sample_indices:
            recommendations = self.recommend_similar_users(idx, n_recommendations)
            
            user_cluster = self.user_clusters[idx]
            same_cluster_count = sum(
                1 for rec in recommendations 
                if rec['cluster'] == user_cluster
            )
            intra_cluster_ratios.append(same_cluster_count / len(recommendations))
            
            avg_similarity = np.mean([rec['similarity'] for rec in recommendations])
            similarity_scores.append(avg_similarity)
        
        avg_intra_cluster = np.mean(intra_cluster_ratios)
        avg_similarity = np.mean(similarity_scores)
        
        results = {
            'avg_intra_cluster_ratio': avg_intra_cluster,
            'avg_similarity_score': avg_similarity,
            'samples_tested': len(sample_indices)
        }
        
        print(f"\n推荐相似度指标:")
        print(f"  平均同簇推荐比例: {avg_intra_cluster:.2%}")
        print(f"  平均余弦相似度: {avg_similarity:.4f}")
        
        # 评分
        score = 0
        if avg_intra_cluster > 0.6:
            score += 40
        elif avg_intra_cluster > 0.4:
            score += 25
        
        if avg_similarity > 0.7:
            score += 35
        elif avg_similarity > 0.5:
            score += 20
        
        results['similarity_score'] = score
        print(f"\n推荐相似度总分: {score}/75")
        
        self.validation_results['recommendation_similarity'] = results
        return results

    def cross_validation_stability(self, n_splits=5):
        """验证3: 聚类稳定性"""
        print("\n" + "="*60)
        print("验证3: 聚类稳定性验证")
        print("="*60)
        
        features = self.genuine_features
        n_samples = len(features)
        indices = np.arange(n_samples)
        split_size = n_samples // n_splits
        
        stability_scores = []
        
        for i in range(n_splits):
            np.random.shuffle(indices)
            subset = indices[:split_size]
            
            kmeans1 = KMeans(
                n_clusters=self.kmeans.n_clusters,
                random_state=self.config.random_state + i,
                n_init=10
            )
            kmeans2 = KMeans(
                n_clusters=self.kmeans.n_clusters,
                random_state=self.config.random_state + i + 100,
                n_init=10
            )
            
            labels1 = kmeans1.fit_predict(features[subset])
            labels2 = kmeans2.fit_predict(features[subset])
            
            ari = adjusted_rand_score(labels1, labels2)
            stability_scores.append(ari)
        
        avg_stability = np.mean(stability_scores)
        
        results = {
            'avg_stability': avg_stability,
            'stability_scores': stability_scores
        }
        
        print(f"\n平均ARI系数: {avg_stability:.4f} ([0,1], 越高越稳定)")
        
        score = 0
        if avg_stability > 0.7:
            score += 25
        elif avg_stability > 0.5:
            score += 15
        
        results['stability_score'] = score
        print(f"聚类稳定性总分: {score}/25")
        
        self.validation_results['stability'] = results
        return results

    def run_full_validation(self):
        """运行完整验证流程（移除多样性验证）"""
        print("\n" + "="*60)
        print("开始完整验证流程")
        print("="*60)
        
        self.validate_clustering_quality()
        self.validate_recommendation_similarity(n_samples=100)
        self.cross_validation_stability(n_splits=5)
        
        self.visualize_validation_results()
        self.generate_validation_report()
        
        print("\n验证完成！")

    def visualize_validation_results(self):
        """可视化验证结果（移除多样性相关）"""
        print("\n生成验证结果可视化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 聚类质量
        ax = axes[0, 0]
        clustering = self.validation_results['clustering_quality']
        metrics = {
            '轮廓系数': clustering['silhouette_score'],
            'DB指数(归一化)': 1/(clustering['davies_bouldin_index'] + 1e-8),
            '簇均衡度': clustering['size_balance_score']
        }
        ax.bar(metrics.keys(), metrics.values(), color=['#3498db', '#e74c3c', '#2ecc71'])
        ax.set_ylabel('归一化分数')
        ax.set_title('聚类质量指标')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # 2. 推荐相似度
        ax = axes[0, 1]
        similarity = self.validation_results['recommendation_similarity']
        metrics = {
            '同簇推荐比例': similarity['avg_intra_cluster_ratio'],
            '平均相似度': similarity['avg_similarity_score']
        }
        ax.bar(metrics.keys(), metrics.values(), color=['#9b59b6', '#f39c12'])
        ax.set_ylabel('分数')
        ax.set_title('推荐相似度指标')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # 3. 综合评分
        ax = axes[1, 0]
        total_scores = {
            '聚类质量': self.validation_results['clustering_quality']['clustering_score'],
            '推荐相似度': self.validation_results['recommendation_similarity']['similarity_score'],
            '聚类稳定性': self.validation_results['stability']['stability_score']
        }
        colors = ['#3498db', '#e74c3c', '#9b59b6']
        bars = ax.barh(list(total_scores.keys()), list(total_scores.values()), color=colors)
        ax.set_xlabel('得分')
        ax.set_title('各维度评分')
        ax.set_xlim([0, 80])
        # 添加数值标签
        for bar, score in zip(bars, total_scores.values()):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{score}', va='center')
        
        # 4. 雷达图
        ax = axes[1, 1]
        ax.remove()
        ax = fig.add_subplot(224, projection='polar')
        
        categories = list(total_scores.keys())
        values = list(total_scores.values())
        max_scores = [75, 75, 25]  # 对应聚类质量、推荐相似度、稳定性
        normalized_values = [v/m*100 for v, m in zip(values, max_scores)]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        normalized_values += normalized_values[:1]
        angles += angles[:1]
        
        ax.plot(angles, normalized_values, 'o-', linewidth=2, color='#3498db')
        ax.fill(angles, normalized_values, alpha=0.25, color='#3498db')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('综合评分雷达图', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        save_path = f"{self.config.RESULT_PATH}/validation_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 验证结果可视化已保存: {save_path}")

    def generate_validation_report(self):
        """生成验证报告（移除多样性相关）"""
        report_path = f"{self.config.RESULT_PATH}/recommendation_validation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("个性化推荐系统效果验证报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"数据概览:\n")
            f.write(f"  真实用户数: {len(self.genuine_df)}\n")
            f.write(f"  特征维度: {self.genuine_features.shape[1]}\n")
            f.write(f"  聚类数: {len(np.unique(self.user_clusters))}\n\n")
            
            clustering = self.validation_results['clustering_quality']
            similarity = self.validation_results['recommendation_similarity']
            stability = self.validation_results['stability']
            
            f.write("1. 聚类质量验证\n")
            f.write("-"*70 + "\n")
            f.write(f"轮廓系数: {clustering['silhouette_score']:.4f}\n")
            f.write(f"Davies-Bouldin指数: {clustering['davies_bouldin_index']:.4f}\n")
            f.write(f"簇大小均衡度: {clustering['size_balance_score']:.4f}\n")
            f.write(f"评分: {clustering['clustering_score']}/75\n\n")
            
            f.write("2. 推荐相似度验证\n")
            f.write("-"*70 + "\n")
            f.write(f"平均同簇推荐比例: {similarity['avg_intra_cluster_ratio']:.2%}\n")
            f.write(f"平均余弦相似度: {similarity['avg_similarity_score']:.4f}\n")
            f.write(f"评分: {similarity['similarity_score']}/75\n\n")
            
            f.write("3. 聚类稳定性验证\n")
            f.write("-"*70 + "\n")
            f.write(f"平均ARI系数: {stability['avg_stability']:.4f}\n")
            f.write(f"评分: {stability['stability_score']}/25\n\n")
            
            total_score = sum([
                clustering['clustering_score'],
                similarity['similarity_score'],
                stability['stability_score']
            ])
            
            f.write("="*70 + "\n")
            f.write(f"综合总分: {total_score}/175\n")  # 75+75+25=175
            f.write(f"综合评级: {self._get_grade(total_score)}\n")
            f.write("="*70 + "\n")
        
        print(f"✓ 验证报告已保存: {report_path}")
        print(f"\n最终综合评分: {total_score}/175")
        print(f"综合评级: {self._get_grade(total_score)}")

    def _get_grade(self, score):
        """评级（适配新的总分175）"""
        if score >= 140:  # 80%
            return "优秀 (A)"
        elif score >= 114:  # 65%
            return "良好 (B)"
        elif score >= 88:  # 50%
            return "中等 (C)"
        else:
            return "需改进 (D)"

    def save_recommendation_model(self):
        """保存推荐模型"""
        model_data = {
            'kmeans': self.kmeans,
            'knn_model': self.knn_model,
            'cluster_profiles': self.cluster_profiles,
            'scaler': self.scaler,
            'validation_results': self.validation_results,
            'user_clusters': self.user_clusters
        }
        
        save_path = f"{self.config.MODEL_PATH}/user_recommendation_model.pkl"
        joblib.dump(model_data, save_path)
        print(f"\n✓ 推荐系统模型已保存: {save_path}")

    # ==================== 新增：推荐示例展示 ====================
    def show_recommendation_examples(self, n_examples=3, n_recommendations=5):
        """展示推荐示例"""
        print("\n" + "="*60)
        print("推荐结果示例")
        print("="*60)
        
        sample_indices = np.random.choice(len(self.genuine_df), n_examples, replace=False)
        
        for idx in sample_indices:
            user_info = self.genuine_df.iloc[idx]
            recommendations = self.recommend_similar_users(idx, n_recommendations)
            
            print(f"\n【目标用户 {idx}】")
            if 'screen_name' in user_info:
                print(f"  用户名: {user_info['screen_name']}")
            if 'followers_count' in user_info:
                print(f"  粉丝数: {user_info['followers_count']}")
            print(f"  所属簇: {self.user_clusters[idx]}")
            
            print(f"  推荐相似用户:")
            for i, rec in enumerate(recommendations, 1):
                print(f"    {i}. 用户{rec['user_index']} | 相似度: {rec['similarity']:.3f} | 簇: {rec['cluster']}")

# ==================== 主函数（独立运行入口） ====================
def main():
    """独立运行主函数"""
    print("="*70)
    print("个性化推荐系统 - 独立运行版本")
    print("流程：Bot过滤 → 聚类 → 推荐 → 效果验证（移除多样性验证）")
    print("="*70)
    
    try:
        # 1. 初始化配置
        config = Config()
        print(f"\n配置信息:")
        print(f"  数据目录: {config.output_dir}")
        print(f"  模型目录: {config.model_dir}")
        
        # 2. 初始化推荐系统
        rec_system = UserRecommendationSystem(config)
        
        # 3. 第一步：Bot检测过滤真实用户
        rec_system.load_bot_detection_model()
        
        # 4. 加载过滤后的真实用户数据
        rec_system.load_genuine_users()
        
        # 5. 执行用户聚类（自动生成标签）
        rec_system.cluster_users()
        
        # 6. 可视化聚类结果
        rec_system.visualize_clusters()
        
        # 7. 分析簇画像
        rec_system.analyze_cluster_profiles()
        
        # 8. 构建KNN推荐模型
        rec_system.build_knn_recommender(n_neighbors=10)
        
        # 9. 运行完整效果验证（移除多样性）
        rec_system.run_full_validation()
        
        # 10. 展示推荐示例
        rec_system.show_recommendation_examples(n_examples=3)
        
        # 11. 保存模型
        rec_system.save_recommendation_model()
        
        print("\n" + "="*70)
        print("✓ 个性化推荐系统运行完成！")
        print("="*70)
        print(f"\n生成文件列表:")
        print(f"  1. {config.RESULT_PATH}/cluster_evaluation.png")
        print(f"  2. {config.RESULT_PATH}/user_clusters_visualization.png")
        print(f"  3. {config.RESULT_PATH}/validation_results.png")
        print(f"  4. {config.RESULT_PATH}/recommendation_validation_report.txt")
        print(f"  5. {config.RESULT_PATH}/recommendation_clusters.npy")
        print(f"  6. {config.MODEL_PATH}/user_recommendation_model.pkl")
        
    except Exception as e:
        print(f"\n❌ 程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()