import torch
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import gc


class ColdStartRecommender:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载训练好的模型（适配新版DomainAdaptiveDeBERTa）
        self.model = self._load_trained_model()
        self.tokenizer = self._load_tokenizer()
        
        # 画像数据相关（核心修改：优先加载UserProfiler的增强版数据）
        self.user_profiles = None
        self.user_embeddings = None
        self.actual_cluster_num = 0  # 动态记录实际聚类数量
        self.cluster_mapping = {}    # 映射KNN索引到实际cluster ID
        self._load_or_build_user_profiles()
        
        # 构建原型
        self.prototypes = None
        self.knn_model = None
        
    def _load_tokenizer(self):
        """加载DeBERTaV2分词器（适配新版模型）"""
        from transformers import DebertaV2Tokenizer
        tokenizer = DebertaV2Tokenizer.from_pretrained(self.config.DEBERTA_PATH)
        return tokenizer
    
    def _load_trained_model(self):
        """加载新版DomainAdaptiveDeBERTa模型"""
        # 导入新版模型类
        from domain_adaptation import DomainAdaptiveDeBERTa
        
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, 'best_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到训练好的模型: {model_path}")
        
        # 加载模型权重（适配新版保存格式）
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model = DomainAdaptiveDeBERTa(self.config).to(self.device)
        
        # 兼容不同保存格式
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        
        # 关闭GRL（推理阶段无需梯度反转）
        model.grl.set_alpha(0.0)
        
        print(f"✓ 加载新版模型: {model_path}")
        print(f"✓ 模型设备: {self.device}")
        return model
    
    def _generate_user_embeddings(self):
        """生成用户embedding（适配新版模型多任务输出）"""
        print("\n=== 生成用户画像embedding ===")
        
        # 加载全量数据
        if os.path.exists(os.path.join(self.config.OUTPUT_PATH, "val_set.csv")):
            df = pd.read_csv(os.path.join(self.config.OUTPUT_PATH, "val_set.csv"))
        else:
            # 加载原始预处理数据
            cresci_df = pd.read_csv(self.config.PREPROCESSED_CRESCI)
            gender_df = pd.read_csv(self.config.PREPROCESSED_GENDER)
            df = pd.concat([cresci_df, gender_df], ignore_index=True)
        
        # 数据过滤
        df = df[df['text'].str.len() >= self.config.MIN_TEXT_LENGTH].reset_index(drop=True)
        df = df.sample(n=min(5000, len(df)), random_state=42)  # 采样减少计算量
        
        # 生成embedding
        embeddings = []
        user_ids = []
        texts = []
        domains = []
        labels = []
        
        # 分批处理
        batch_size = 32
        for i in tqdm(range(0, len(df), batch_size), desc="生成embedding"):
            batch_df = df.iloc[i:i+batch_size]
            
            # 编码文本
            encoded = self.tokenizer(
                batch_df['text'].tolist(),
                padding=True,
                truncation=True,
                max_length=self.config.MAX_LENGTH,
                return_tensors='pt'
            ).to(self.device)
            
            # 前向推理（适配新版模型输出）
            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    domain=batch_df['domain'].tolist()  # 传入domain列表
                )
                batch_embeddings = outputs['features'].cpu().numpy()
            
            # 收集结果
            embeddings.extend(batch_embeddings)
            user_ids.extend(batch_df.get('user_id', range(i, i+len(batch_df))).tolist())
            texts.extend(batch_df['text'].tolist())
            domains.extend(batch_df['domain'].tolist())
            labels.extend(batch_df['label'].tolist())
            
            # 内存清理
            del encoded, outputs, batch_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # 转换为数组
        self.user_embeddings = np.array(embeddings)
        
        # 聚类生成cluster标签（核心修改：动态适配聚类数量）
        print("\n=== 对用户embedding进行KMeans聚类 ===")
        # 先尝试加载UserProfiler的聚类指标
        metrics_path = os.path.join(self.config.OUTPUT_PATH, 'clustering_metrics_detailed.txt')
        if os.path.exists(metrics_path):
            # 读取UserProfiler选择的最优K值
            with open(metrics_path, 'r', encoding='utf-8') as f:
                content = f.read()
                import re
                k_match = re.search(r'聚类数量: (\d+)', content)
                if k_match:
                    self.actual_cluster_num = int(k_match.group(1))
                    print(f"✓ 从UserProfiler读取最优聚类数量: {self.actual_cluster_num}")
        
        # 若未读取到，则使用配置值（降级策略）
        if self.actual_cluster_num <= 1:
            self.actual_cluster_num = self.config.NUM_CLUSTERS
            print(f"⚠️  未找到UserProfiler聚类结果，使用配置值: {self.actual_cluster_num}")
        
        # 执行聚类
        kmeans = KMeans(n_clusters=self.actual_cluster_num, random_state=42)
        clusters = kmeans.fit_predict(self.user_embeddings)
        
        # 构建用户画像数据
        self.user_profiles = pd.DataFrame({
            'user_id': user_ids,
            'text': texts,
            'domain': domains,
            'label': labels,
            'cluster': clusters
        })
        
        # 保存画像数据
        os.makedirs(self.config.OUTPUT_PATH, exist_ok=True)
        self.user_profiles.to_csv(os.path.join(self.config.OUTPUT_PATH, 'user_profiles.csv'), index=False)
        np.save(os.path.join(self.config.OUTPUT_PATH, 'user_embeddings.npy'), self.user_embeddings)
        
        print(f"✓ 生成 {len(self.user_profiles)} 条用户画像")
        print(f"✓ 实际聚类数量: {self.actual_cluster_num}")
        print(f"✓ Cluster分布: {pd.Series(clusters).value_counts().to_dict()}")
    
    def _load_or_build_user_profiles(self):
        """核心修改：优先加载UserProfiler的增强版数据"""
        # 优先加载UserProfiler生成的增强版数据
        profile_paths = [
            os.path.join(self.config.OUTPUT_PATH, 'user_profiles_enhanced.csv'),
            os.path.join(self.config.OUTPUT_PATH, 'user_profiles.csv')
        ]
        embedding_path = os.path.join(self.config.OUTPUT_PATH, 'user_embeddings.npy')
        
        # 寻找可用的画像文件
        profile_path = None
        for p in profile_paths:
            if os.path.exists(p):
                profile_path = p
                break
        
        if profile_path and os.path.exists(embedding_path):
            # 加载已有数据（核心：动态获取实际聚类数量）
            self.user_profiles = pd.read_csv(profile_path)
            self.user_embeddings = np.load(embedding_path)
            
            # 动态统计实际聚类数量和ID
            actual_clusters = sorted(self.user_profiles['cluster'].unique())
            self.actual_cluster_num = len(actual_clusters)
            # 构建索引映射（KNN索引 → 实际cluster ID）
            self.cluster_mapping = {i: cid for i, cid in enumerate(actual_clusters)}
            
            print(f"✓ 加载已有用户画像: {len(self.user_profiles)} 条")
            print(f"✓ 实际聚类数量: {self.actual_cluster_num} (cluster ID: {actual_clusters})")
        else:
            # 生成新数据
            self._generate_user_embeddings()
            # 生成后重新统计
            actual_clusters = sorted(self.user_profiles['cluster'].unique())
            self.cluster_mapping = {i: cid for i, cid in enumerate(actual_clusters)}
    
    def build_prototypes(self):
        """核心修改：动态适配实际聚类数量，跳过空cluster"""
        print("\n=== 构建Cluster原型 ===")
        print(f"📌 适配实际聚类数量: {self.actual_cluster_num}")
        
        self.prototypes = {}
        prototype_domains = {}  # 记录每个cluster的主要域
        
        # 核心修改：遍历实际存在的cluster ID，而非配置的NUM_CLUSTERS
        actual_clusters = sorted(self.user_profiles['cluster'].unique())
        for cluster_id in actual_clusters:
            cluster_mask = self.user_profiles['cluster'] == cluster_id
            cluster_embeddings = self.user_embeddings[cluster_mask]
            cluster_domains = self.user_profiles.loc[cluster_mask, 'domain']
            
            if len(cluster_embeddings) == 0:
                print(f"⚠️ Cluster {cluster_id} 无数据，跳过")
                continue
            
            # 原型 = cluster的均值向量
            prototype = cluster_embeddings.mean(axis=0)
            self.prototypes[cluster_id] = prototype
            
            # 统计cluster的域分布
            domain_dist = cluster_domains.value_counts()
            main_domain = domain_dist.index[0] if len(domain_dist) > 0 else 'unknown'
            prototype_domains[cluster_id] = main_domain
            
            print(f"Cluster {cluster_id}: {cluster_mask.sum()} 用户 | 主要域: {main_domain} | 原型向量 {prototype.shape}")
        
        # 构建KNN模型（核心：基于实际有效cluster）
        valid_clusters = list(self.prototypes.keys())
        if not valid_clusters:
            raise ValueError("无有效cluster，无法构建KNN模型")
        
        # 更新映射关系（KNN索引 → 实际cluster ID）
        self.cluster_mapping = {i: cid for i, cid in enumerate(valid_clusters)}
        prototype_matrix = np.vstack([self.prototypes[i] for i in valid_clusters])
        
        self.knn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
        self.knn_model.fit(prototype_matrix)
        
        # 保存cluster-域映射
        self.prototype_domains = prototype_domains
        print(f"✓ 原型构建完成（有效cluster: {len(valid_clusters)}），KNN模型就绪")
        print(f"✓ KNN索引映射: {self.cluster_mapping}")
    
    def predict_cold_start_user(self, user_texts, domain='cresci'):
        """核心修改：修复KNN索引到实际cluster ID的映射"""
        # 提取用户embedding
        with torch.no_grad():
            encoded = self.tokenizer(
                user_texts,
                padding=True,
                truncation=True,
                max_length=self.config.MAX_LENGTH,
                return_tensors='pt'
            ).to(self.device)
            
            # 适配新版模型：传入domain参数
            outputs = self.model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                domain=[domain] * len(user_texts)
            )
            
            # 取多条推文的平均embedding
            user_embedding = outputs['features'].cpu().numpy().mean(axis=0, keepdims=True)
        
        # KNN检索最近的原型（核心：使用映射关系转换索引）
        distances, indices = self.knn_model.kneighbors(user_embedding)
        
        # 关键修复：将KNN返回的索引转换为实际cluster ID
        predicted_index = indices[0][0]
        predicted_cluster = self.cluster_mapping.get(predicted_index, -1)
        main_domain = self.prototype_domains.get(predicted_cluster, 'unknown')
        
        if predicted_cluster == -1:
            print(f"⚠️  预测索引 {predicted_index} 无对应cluster ID")
        
        return predicted_cluster, distances[0][0], main_domain
    
    def evaluate_cold_start(self):
        """评估冷启动效果（适配动态聚类数量）"""
        print("\n=== 评估冷启动效果 ===")
        
        # 模拟冷启动场景：随机采样用户，只用前N条推文
        test_sizes = [1, 3, 5, 10]
        results = {
            'overall': {},
            'cresci': {},
            'gender': {}
        }
        
        # 采样测试用户（按域分层）
        test_users = self.user_profiles.groupby('domain').apply(
            lambda x: x.sample(n=min(100, len(x)), random_state=42)
        ).reset_index(drop=True)
        
        for n_tweets in test_sizes:
            print(f"\n使用前 {n_tweets} 条推文...")
            
            # 按域统计预测结果
            predictions = {'overall': [], 'cresci': [], 'gender': []}
            true_labels = {'overall': [], 'cresci': [], 'gender': []}
            
            for idx, row in tqdm(test_users.iterrows(), total=len(test_users), desc=f"测试"):
                # 模拟：只取文本的前n_tweets个句子
                text = row['text']
                sentences = text.split('.')[:n_tweets]
                truncated_text = ['. '.join(sentences)]
                
                # 预测（传入用户所属域）
                pred_cluster, _, _ = self.predict_cold_start_user(
                    truncated_text, 
                    domain=row['domain']
                )
                
                # 过滤无效预测
                if pred_cluster == -1:
                    continue
                
                # 按域记录结果
                domain = row['domain']
                predictions['overall'].append(pred_cluster)
                true_labels['overall'].append(row['cluster'])
                
                if domain in predictions:
                    predictions[domain].append(pred_cluster)
                    true_labels[domain].append(row['cluster'])
            
            # 计算各域准确率（处理空列表）
            def safe_accuracy(true, pred):
                if len(true) == 0 or len(pred) == 0:
                    return 0.0
                return accuracy_score(true, pred)
            
            results['overall'][n_tweets] = safe_accuracy(true_labels['overall'], predictions['overall'])
            results['cresci'][n_tweets] = safe_accuracy(true_labels['cresci'], predictions['cresci'])
            results['gender'][n_tweets] = safe_accuracy(true_labels['gender'], predictions['gender'])
            
            # 打印结果
            print(f"整体准确率: {results['overall'][n_tweets]:.4f}")
            if 'cresci' in results and n_tweets in results['cresci']:
                print(f"Cresci域准确率: {results['cresci'][n_tweets]:.4f}")
            if 'gender' in results and n_tweets in results['gender']:
                print(f"Gender域准确率: {results['gender'][n_tweets]:.4f}")
        
        # 可视化冷启动效果（新增多域对比）
        self.plot_cold_start_results(results)
        
        return results
    
    def plot_cold_start_results(self, results):
        """绘制冷启动效果曲线（多域对比）"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 绘制不同域的曲线
        colors = {'overall': '#2E86AB', 'cresci': '#A23B72', 'gender': '#F18F01'}
        markers = {'overall': 'o', 'cresci': 's', 'gender': '^'}
        
        for domain, color in colors.items():
            if domain in results and results[domain]:
                x = list(results[domain].keys())
                y = list(results[domain].values())
                
                ax.plot(x, y, 
                        marker=markers[domain], 
                        linewidth=2, 
                        markersize=10, 
                        color=color,
                        label=f'{domain.capitalize()} Domain')
                ax.fill_between(x, y, alpha=0.2, color=color)
                
                # 添加数值标签
                for xi, yi in zip(x, y):
                    ax.text(xi, yi + 0.02, f'{yi:.2%}', ha='center', fontsize=9, color=color)
        
        # 核心修改：标题显示实际聚类数量
        ax.set_xlabel('Number of Tweets', fontsize=12)
        ax.set_ylabel('User Profile Prediction Accuracy', fontsize=12)
        ax.set_title(f'Cold-start Performance (K={self.actual_cluster_num})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.OUTPUT_PATH, 'cold_start_performance.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 冷启动效果图: {save_path}")
        plt.close()
    
    def demo_recommendation(self):
        """演示：基于画像推荐相似用户（适配动态聚类数量）"""
        print("\n=== 推荐系统Demo（多域适配）===")
        
        # 按域分层选择示例用户
        for domain in ['cresci', 'gender']:
            domain_users = self.user_profiles[self.user_profiles['domain'] == domain]
            if len(domain_users) == 0:
                continue
            
            # 随机选择一个用户
            sample_user = domain_users.sample(n=1).iloc[0]
            user_idx = sample_user.name
            user_cluster = sample_user['cluster']
            user_embedding = self.user_embeddings[user_idx:user_idx+1]
            
            print(f"\n【{domain.upper()} 域示例】")
            print(f"目标用户: {sample_user['user_id']}")
            print(f"所属Cluster: {user_cluster} (实际聚类数量: {self.actual_cluster_num})")
            print(f"文本片段: {sample_user['text'][:100]}...")
            
            # 推荐同cluster的相似用户（同域优先）
            cluster_users = self.user_profiles[
                (self.user_profiles['cluster'] == user_cluster) & 
                (self.user_profiles['domain'] == domain)
            ]
            
            if len(cluster_users) < 6:  # 不足则放宽域限制
                cluster_users = self.user_profiles[self.user_profiles['cluster'] == user_cluster]
            
            if len(cluster_users) < 2:
                print(f"⚠️  Cluster {user_cluster} 相似用户不足，跳过")
                continue
            
            cluster_indices = cluster_users.index.tolist()
            cluster_embeddings = self.user_embeddings[cluster_indices]
            
            # 计算余弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(user_embedding, cluster_embeddings)[0]
            
            # Top 5推荐（排除自己）
            top_indices = np.argsort(similarities)[::-1][1:6]
            if len(top_indices) < 1:
                print(f"⚠️  相似用户不足，跳过")
                continue
            
            recommended_users = cluster_users.iloc[top_indices]
            
            print(f"推荐的相似用户 (Top {len(top_indices)}):")
            for i, (idx, row) in enumerate(recommended_users.iterrows(), 1):
                print(f"{i}. User {row['user_id']} (相似度: {similarities[top_indices[i-1]]:.4f})")
                print(f"   域: {row['domain']} | 标签: {row['label']}")
                print(f"   文本: {row['text'][:80]}...")
                print("   ---")
    
    def run(self):
        """执行完整冷启动流程"""
        print("=" * 60)
        print("步骤4: 冷启动原型学习（适配动态聚类数量）")
        print("=" * 60)
        print(f"📌 关键配置:")
        print(f"   - 配置聚类数量: {self.config.NUM_CLUSTERS}")
        print(f"   - 实际聚类数量: {self.actual_cluster_num}")
        
        # 构建原型
        self.build_prototypes()
        
        # 评估冷启动
        results = self.evaluate_cold_start()
        
        # Demo推荐
        self.demo_recommendation()
        
        print("\n" + "=" * 60)
        print("✅ 步骤4完成!")
        print("=" * 60)
        
        return results


if __name__ == "__main__":
    from config import Config
    
    config = Config()
    # 补充冷启动相关配置（若Config中未定义）
    if not hasattr(config, 'NUM_CLUSTERS'):
        config.NUM_CLUSTERS = 8  # 仅作为降级默认值
    if not hasattr(config, 'MAX_LENGTH'):
        config.MAX_LENGTH = 512
    if not hasattr(config, 'MIN_TEXT_LENGTH'):
        config.MIN_TEXT_LENGTH = 10
    
    recommender = ColdStartRecommender(config)
    results = recommender.run()