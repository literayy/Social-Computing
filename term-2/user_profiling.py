import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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

class DomainAdaptiveDeBERTa(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        from transformers import DebertaV2Model
        self.deberta = DebertaV2Model.from_pretrained(config.DEBERTA_PATH)
        self.hidden_size = self.deberta.config.hidden_size
        print(f"✓ DeBERTa hidden_size: {self.hidden_size}")
        
        # 初始化GRL（使用config中的参数）
        self.grl = GradientReversalLayer(alpha=config.GRL_ALPHA)
        
        # 增强版域分类器（与训练代码一致）
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, config.NUM_DOMAINS)
        )
        
        # Bot分类器
        self.bot_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, config.NUM_BOT_CLASSES)
        )
        
        # 增强版性别分类器（与训练代码一致）
        self.gender_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
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

# ==================== 用户画像主类（优化版） ====================
class UserProfiler:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device_type)
        print(f"使用设备: {self.device}")
        
        # 加载模型和Tokenizer
        self.model = self._load_trained_model()
        self.tokenizer = self._load_tokenizer()
        
        # 初始化变量
        self.user_embeddings = None
        self.user_df = None
        self.cluster_labels = None
        self.cluster_profiles = None
        self.cluster_metrics = {}  # 存储多维度聚类指标
        
        # 优化参数（可根据结果调整）
        self.OPTIMIZE_NUM_CLUSTERS = True  # 自动优化聚类数量
        self.CLUSTER_RANGE = [4, 6, 8, 10]  # 候选聚类数量
        self.USE_HIERARCHICAL_CLUSTERING = False  # 层级聚类（可选）
        self.FEATURE_SCALING = True  # 特征标准化
        
    def _load_tokenizer(self):
        from transformers import DebertaV2Tokenizer
        try:
            tokenizer = DebertaV2Tokenizer.from_pretrained(self.config.DEBERTA_PATH)
            print("✓ Tokenizer加载成功")
            return tokenizer
        except Exception as e:
            raise Exception(f"Tokenizer加载失败: {e}\n请检查DEBERTA_PATH配置: {self.config.DEBERTA_PATH}")
    
    def _load_trained_model(self):
        print("\n=== 加载训练模型 ===")
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, 'best_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}\n请检查MODEL_SAVE_PATH配置: {self.config.MODEL_SAVE_PATH}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        model = DomainAdaptiveDeBERTa(self.config).to(self.device)
        
        # 容错加载权重
        def load_state_dict_with_adjustment(model, state_dict):
            model_dict = model.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"⚠️  跳过不匹配的权重: {k}")
            model_dict.update(filtered_state_dict)
            model.load_state_dict(model_dict)
            print(f"✓ 成功加载 {len(filtered_state_dict)}/{len(state_dict)} 个匹配的权重参数")
        
        load_state_dict_with_adjustment(model, state_dict)
        model.eval()
        print(f"✓ 模型加载完成: {model_path}")
        return model
    
    def extract_user_embeddings(self):
        print("\n=== 提取用户embeddings ===")
        
        # 加载预处理数据
        def load_data(file_path, desc):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{desc}数据文件不存在: {file_path}")
            df = pd.read_csv(file_path)
            df = df[df['text'].str.len() >= self.config.MIN_TEXT_LENGTH].reset_index(drop=True)
            print(f"✓ {desc}数据加载完成: {len(df)} 条")
            
            if len(df) > self.config.PROFILING_SAMPLE_SIZE:
                df = df.sample(n=self.config.PROFILING_SAMPLE_SIZE, random_state=self.config.RANDOM_SEED)
                print(f"✓ {desc}数据采样至: {len(df)} 条")
            return df
        
        cresci_df = load_data(self.config.PREPROCESSED_CRESCI, "Cresci")
        gender_df = load_data(self.config.PREPROCESSED_GENDER, "Gender")
        
        # 添加域标签
        cresci_df['domain'] = 'cresci'
        gender_df['domain'] = 'gender'
        all_df = pd.concat([cresci_df, gender_df], ignore_index=True)
        print(f"✓ 合并后用户总数: {len(all_df)}")
        
        # 提取embeddings
        embeddings = []
        batch_size = self.config.BATCH_SIZE
        
        with torch.no_grad():
            for i in tqdm(range(0, len(all_df), batch_size), desc="提取embeddings"):
                batch_df = all_df.iloc[i:i+batch_size]
                batch_texts = batch_df['text'].tolist()
                
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.MAX_LENGTH,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    domain=batch_df['domain'].tolist()
                )
                
                batch_embeddings = outputs['features'].cpu().numpy()
                embeddings.append(batch_embeddings)
                
                del encoded, outputs
                if self.config.device_type == 'cuda':
                    torch.cuda.empty_cache()
        
        self.user_embeddings = np.vstack(embeddings)
        self.user_df = all_df.iloc[:len(self.user_embeddings)].reset_index(drop=True)
        print(f"✓ Embeddings提取完成，shape: {self.user_embeddings.shape}")
        return self.user_embeddings
    
    def _select_best_cluster_num(self, embeddings_pca):
        """自动选择最优聚类数量（基于轮廓系数+DBI）"""
        print("\n=== 自动优化聚类数量 ===")
        best_metrics = {
            'n_clusters': self.config.NUM_CLUSTERS,
            'silhouette': -1,
            'davies_bouldin': 999,
            'calinski_harabasz': 0
        }
        
        for n_clusters in self.CLUSTER_RANGE:
            if n_clusters >= len(embeddings_pca):
                continue
            
            # 训练K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.RANDOM_SEED, n_init=10)
            labels = kmeans.fit_predict(embeddings_pca)
            
            # 计算多维度指标
            try:
                silhouette = silhouette_score(embeddings_pca, labels)
                davies_bouldin = davies_bouldin_score(embeddings_pca, labels)
                calinski_harabasz = calinski_harabasz_score(embeddings_pca, labels)
                
                print(f"K={n_clusters}: 轮廓系数={silhouette:.4f}, DBI={davies_bouldin:.4f}, CH指数={calinski_harabasz:.2f}")
                
                # 综合评分（轮廓系数越高+DBI越低越好）
                score = silhouette - (davies_bouldin / 10)  # 归一化权重
                
                # 更新最优值
                if score > (best_metrics['silhouette'] - (best_metrics['davies_bouldin'] / 10)):
                    best_metrics = {
                        'n_clusters': n_clusters,
                        'silhouette': silhouette,
                        'davies_bouldin': davies_bouldin,
                        'calinski_harabasz': calinski_harabasz,
                        'labels': labels
                    }
            except:
                continue
        
        print(f"\n✅ 最优聚类数量: K={best_metrics['n_clusters']}")
        print(f"   最优指标: 轮廓系数={best_metrics['silhouette']:.4f}, DBI={best_metrics['davies_bouldin']:.4f}")
        return best_metrics['n_clusters'], best_metrics['labels'], best_metrics
    
    def perform_clustering(self):
        """优化版聚类流程"""
        print("\n=== 层次化聚类（优化版） ===")
        
        if self.user_embeddings is None:
            raise ValueError("请先提取用户embeddings")
        
        # 步骤1: 特征标准化（提升聚类效果）
        embeddings_scaled = self.user_embeddings
        if self.FEATURE_SCALING:
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(self.user_embeddings)
            print("✓ 特征标准化完成")
        
        # 步骤2: PCA降维（优化维度选择）
        n_components = min(50, embeddings_scaled.shape[1], len(embeddings_scaled)-1)
        if n_components < 2:
            n_components = 2
        
        pca = PCA(n_components=n_components, random_state=self.config.RANDOM_SEED)
        embeddings_pca = pca.fit_transform(embeddings_scaled)
        print(f"✓ PCA降维完成: 保留维度={embeddings_pca.shape[1]}, 方差保留={pca.explained_variance_ratio_.sum():.2%}")
        
        # 步骤3: 选择聚类数量并执行聚类
        if self.OPTIMIZE_NUM_CLUSTERS:
            n_clusters, self.cluster_labels, self.cluster_metrics = self._select_best_cluster_num(embeddings_pca)
        else:
            n_clusters = min(self.config.NUM_CLUSTERS, len(embeddings_pca))
            if n_clusters < 2:
                n_clusters = 2
            
            # 选择聚类算法
            if self.USE_HIERARCHICAL_CLUSTERING:
                # 层级聚类（更适合用户画像的模糊边界）
                clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                self.cluster_labels = clustering.fit_predict(embeddings_pca)
                print(f"✓ 层级聚类完成 (K={n_clusters})")
            else:
                # K-means聚类（优化n_init参数）
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.RANDOM_SEED, n_init=20)
                self.cluster_labels = kmeans.fit_predict(embeddings_pca)
                print(f"✓ K-means聚类完成 (K={n_clusters}, n_init=20)")
            
            # 计算完整指标
            try:
                self.cluster_metrics['silhouette'] = silhouette_score(embeddings_pca, self.cluster_labels)
            except:
                self.cluster_metrics['silhouette'] = -1
            
            try:
                self.cluster_metrics['davies_bouldin'] = davies_bouldin_score(embeddings_pca, self.cluster_labels)
            except:
                self.cluster_metrics['davies_bouldin'] = 999
            
            try:
                self.cluster_metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings_pca, self.cluster_labels)
            except:
                self.cluster_metrics['calinski_harabasz'] = 0
        
        # 输出最终聚类指标
        print(f"\n📊 最终聚类质量评估:")
        print(f"  轮廓系数: {self.cluster_metrics.get('silhouette', -1):.4f} (越接近1越好)")
        print(f"  Davies-Bouldin指数: {self.cluster_metrics.get('davies_bouldin', 999):.4f} (越小越好)")
        print(f"  Calinski-Harabasz指数: {self.cluster_metrics.get('calinski_harabasz', 0):.2f} (越大越好)")
        
        # 添加聚类标签到用户数据
        self.user_df['cluster'] = self.cluster_labels
        
        # 统计Cluster分布
        cluster_dist = Counter(self.cluster_labels)
        print(f"\n📈 Cluster分布:")
        for cluster_id, count in sorted(cluster_dist.items()):
            print(f"  Cluster {cluster_id}: {count} 用户 ({count/len(self.user_df)*100:.1f}%)")
        
        return self.cluster_labels, self.cluster_metrics
    
    def generate_cluster_profiles(self):
        """增强版Cluster画像生成"""
        print("\n=== 生成Cluster画像（增强版） ===")
        
        if self.cluster_labels is None:
            raise ValueError("请先执行聚类")
        
        self.cluster_profiles = {}
        gender_map = {0: '男性', 1: '女性', 2: '品牌', '0': '男性', '1': '女性', '2': '品牌'}
        
        for cluster_id in sorted(Counter(self.cluster_labels).keys()):
            cluster_users = self.user_df[self.user_df['cluster'] == cluster_id]
            if len(cluster_users) == 0:
                continue
            
            # 基础分布统计
            domain_dist = cluster_users['domain'].value_counts(normalize=True).to_dict()
            label_dist = cluster_users['label'].value_counts(normalize=True).to_dict()
            
            # 增强统计：文本长度分布
            text_lengths = cluster_users['text'].str.len()
            text_stats = {
                'avg_length': text_lengths.mean(),
                'std_length': text_lengths.std(),
                'min_length': text_lengths.min(),
                'max_length': text_lengths.max()
            }
            
            # 关键词提取（优化TF-IDF）
            keywords = []
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(
                    max_features=200,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=2  # 过滤低频词
                )
                tfidf_matrix = vectorizer.fit_transform(cluster_users['text'].fillna(''))
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.sum(axis=0).A1
                top_indices = tfidf_scores.argsort()[-10:][::-1]
                keywords = [(feature_names[i], tfidf_scores[i]) for i in top_indices if tfidf_scores[i] > 0]
            except Exception as e:
                print(f"⚠️  Cluster {cluster_id} 关键词提取失败: {e}")
            
            # 智能命名
            try:
                main_domain = max(domain_dist, key=domain_dist.get)
                if main_domain == 'cresci':
                    main_label = max(label_dist, key=label_dist.get)
                    cluster_type = "Bot用户" if main_label == 1 else "真实用户"
                else:
                    main_label = max(label_dist, key=label_dist.get)
                    cluster_type = f"{gender_map.get(main_label, '未知')}用户"
                
                # 加入规模描述
                size_pct = len(cluster_users) / len(self.user_df) * 100
                cluster_name = f"{cluster_type} (占比{size_pct:.1f}%)"
            except:
                cluster_name = f"未知用户群-{cluster_id}"
            
            self.cluster_profiles[cluster_id] = {
                'name': cluster_name,
                'size': len(cluster_users),
                'size_pct': len(cluster_users) / len(self.user_df) * 100,
                'domain_dist': domain_dist,
                'label_dist': label_dist,
                'text_stats': text_stats,
                'keywords': keywords,
                'sample_users': cluster_users.head(5)['text'].tolist()  # 样本文本
            }
            
            # 输出详细画像
            print(f"\n[Cluster {cluster_id}: {cluster_name}]")
            print(f"  规模: {len(cluster_users)} 用户 ({size_pct:.1f}%)")
            print(f"  域分布: {domain_dist}")
            print(f"  标签分布: {label_dist}")
            print(f"  文本长度: 平均{text_stats['avg_length']:.1f}字 (±{text_stats['std_length']:.1f})")
            print(f"  核心关键词: {', '.join([k[0] for k in keywords[:5]]) if keywords else '无'}")
        
        # 保存增强版画像
        profile_path = os.path.join(self.config.OUTPUT_PATH, 'cluster_profiles_enhanced.csv')
        # 展平字典以便保存
        profile_flat = []
        for cid, profile in self.cluster_profiles.items():
            row = {
                'cluster_id': cid,
                'name': profile['name'],
                'size': profile['size'],
                'size_pct': profile['size_pct'],
                'main_domain': max(profile['domain_dist'], key=profile['domain_dist'].get) if profile['domain_dist'] else '',
                'main_label': max(profile['label_dist'], key=profile['label_dist'].get) if profile['label_dist'] else '',
                'avg_text_length': profile['text_stats']['avg_length'],
                'top_keywords': ', '.join([k[0] for k in profile['keywords'][:5]]) if profile['keywords'] else ''
            }
            profile_flat.append(row)
        
        profile_df = pd.DataFrame(profile_flat)
        profile_df.to_csv(profile_path, encoding='utf-8', index=False)
        print(f"\n✓ 增强版画像保存至: {profile_path}")
        
        return self.cluster_profiles
    
    def visualize_clusters(self):
        """优化版t-SNE可视化"""
        print("\n=== t-SNE可视化（优化版） ===")
        
        if self.user_embeddings is None or self.cluster_labels is None:
            print("⚠️  缺少embeddings或聚类标签，跳过可视化")
            return
        
        # 采样优化
        sample_size = min(1500, len(self.user_embeddings))  # 增加采样数量提升可视化效果
        if sample_size < 10:
            print("⚠️  样本数量过少，跳过可视化")
            return
        
        # 兼容低版本NumPy的随机采样
        np.random.seed(self.config.RANDOM_SEED)
        sample_indices = np.random.choice(len(self.user_embeddings), sample_size, replace=False)
        
        embeddings_sample = self.user_embeddings[sample_indices]
        cluster_sample = self.cluster_labels[sample_indices]
        domain_sample = self.user_df.iloc[sample_indices]['domain'].tolist()
        
        # t-SNE优化（调整参数提升效果）
        try:
            perplexity = min(50, sample_size-1)  # 增大perplexity提升全局结构
            tsne = TSNE(
                n_components=2,
                random_state=self.config.RANDOM_SEED,
                perplexity=perplexity,
                n_iter=2000,  # 增加迭代次数
                learning_rate='auto'
            )
            embeddings_2d = tsne.fit_transform(embeddings_sample)
        except Exception as e:
            print(f"⚠️  t-SNE降维失败: {e}")
            return
        
        # 绘制优化版可视化图
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))  # 增加第三个子图（按标签着色）
        
        # 子图1: 按Cluster着色
        scatter1 = axes[0].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=cluster_sample, cmap='tab10', alpha=0.8, s=40, edgecolors='white', linewidth=0.5
        )
        axes[0].set_title('User Clustering by Cluster ID', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('t-SNE Component 1')
        axes[0].set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter1, ax=axes[0], label='Cluster ID')
        
        # 子图2: 按Domain着色
        domain_colors = [0 if d=='cresci' else 1 for d in domain_sample]
        scatter2 = axes[1].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=domain_colors, cmap='coolwarm', alpha=0.8, s=40, edgecolors='white', linewidth=0.5
        )
        axes[1].set_title('User Clustering by Domain', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('t-SNE Component 1')
        axes[1].set_ylabel('t-SNE Component 2')
        cbar2 = plt.colorbar(scatter2, ax=axes[1], ticks=[0, 1])
        cbar2.set_ticklabels(['Cresci (Bot检测)', 'Gender (性别分类)'])
        
        # 子图3: 按标签着色（Bot/性别）
        label_sample = self.user_df.iloc[sample_indices]['label'].tolist()
        scatter3 = axes[2].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=label_sample, cmap='viridis', alpha=0.8, s=40, edgecolors='white', linewidth=0.5
        )
        axes[2].set_title('User Clustering by Label', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('t-SNE Component 1')
        axes[2].set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter3, ax=axes[2], label='Label (0/1/2)')
        
        plt.suptitle(f'User Clustering Visualization (K={len(Counter(self.cluster_labels))})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.config.OUTPUT_PATH, 'cluster_visualization_enhanced.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✓ 增强版可视化保存至: {save_path}")
    
    def save_results(self):
        """完整结果保存"""
        print("\n=== 保存结果（完整版） ===")
        
        # 1. 用户画像数据
        if self.user_df is not None and len(self.user_df) > 0:
            output_path = os.path.join(self.config.OUTPUT_PATH, 'user_profiles_enhanced.csv')
            self.user_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✓ 用户画像数据: {output_path}")
        
        # 2. Embeddings
        if self.user_embeddings is not None and len(self.user_embeddings) > 0:
            embeddings_path = os.path.join(self.config.OUTPUT_PATH, 'user_embeddings.npy')
            np.save(embeddings_path, self.user_embeddings)
            print(f"✓ 用户embeddings: {embeddings_path}")
        
        # 3. 聚类指标报告
        metrics_path = os.path.join(self.config.OUTPUT_PATH, 'clustering_metrics_detailed.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("聚类质量详细报告\n")
            f.write("="*60 + "\n\n")
            
            f.write("1. 配置参数\n")
            f.write(f"   - 聚类数量: {len(Counter(self.cluster_labels))}\n")
            f.write(f"   - 采样数量: {self.config.PROFILING_SAMPLE_SIZE}\n")
            f.write(f"   - 特征标准化: {self.FEATURE_SCALING}\n")
            f.write(f"   - PCA维度: {min(50, self.user_embeddings.shape[1], len(self.user_embeddings)-1)}\n\n")
            
            f.write("2. 核心指标\n")
            f.write(f"   - 轮廓系数 (Silhouette Score): {self.cluster_metrics.get('silhouette', -1):.4f}\n")
            f.write(f"     (解读: 越接近1越好，0.3+为可接受，0.5+为良好)\n")
            f.write(f"   - Davies-Bouldin指数: {self.cluster_metrics.get('davies_bouldin', 999):.4f}\n")
            f.write(f"     (解读: 越小越好，<1.5为可接受，<1.0为良好)\n")
            f.write(f"   - Calinski-Harabasz指数: {self.cluster_metrics.get('calinski_harabasz', 0):.2f}\n")
            f.write(f"     (解读: 越大越好，数值越高说明聚类越紧凑)\n\n")
            
            f.write("3. Cluster分布\n")
            cluster_dist = Counter(self.cluster_labels)
            total_users = len(self.user_df)
            for cluster_id, count in sorted(cluster_dist.items()):
                pct = count / total_users * 100
                profile = self.cluster_profiles.get(cluster_id, {})
                name = profile.get('name', f"Cluster {cluster_id}")
                f.write(f"   - {name}: {count} 用户 ({pct:.1f}%)\n")
        
        print(f"✓ 聚类指标报告: {metrics_path}")
        
        # 4. 画像摘要
        summary_path = os.path.join(self.config.OUTPUT_PATH, 'cluster_summary.md')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# 用户聚类画像摘要\n")
            f.write("## 聚类质量\n")
            f.write(f"- 轮廓系数: {self.cluster_metrics.get('silhouette', -1):.4f}\n")
            f.write(f"- Davies-Bouldin指数: {self.cluster_metrics.get('davies_bouldin', 999):.4f}\n\n")
            
            f.write("## 各Cluster特征\n")
            for cid, profile in self.cluster_profiles.items():
                f.write(f"### Cluster {cid}: {profile['name']}\n")
                f.write(f"- 规模: {profile['size']} 用户 ({profile['size_pct']:.1f}%)\n")
                f.write(f"- 主要域: {max(profile['domain_dist'], key=profile['domain_dist'].get) if profile['domain_dist'] else '无'}\n")
                f.write(f"- 核心关键词: {', '.join([k[0] for k in profile['keywords'][:5]]) if profile['keywords'] else '无'}\n\n")
        
        print(f"✓ 画像摘要 (Markdown): {summary_path}")
    
    def run(self):
        """执行完整优化流程"""
        print("=" * 60)
        print("步骤3: 层次化用户画像构建（优化版）")
        print("=" * 60)
        print(f"优化配置:")
        print(f"  - 自动优化聚类数量: {self.OPTIMIZE_NUM_CLUSTERS}")
        print(f"  - 候选聚类数量: {self.CLUSTER_RANGE}")
        print(f"  - 特征标准化: {self.FEATURE_SCALING}")
        print(f"  - 层级聚类: {self.USE_HIERARCHICAL_CLUSTERING}")
        
        try:
            # 提取embeddings
            self.extract_user_embeddings()
            
            # 优化聚类
            cluster_labels, metrics = self.perform_clustering()
            
            # 生成增强版画像
            self.generate_cluster_profiles()
            
            # 优化可视化
            self.visualize_clusters()
            
            # 保存完整结果
            self.save_results()
            
            print("\n" + "=" * 60)
            print("✅ 步骤3完成! (优化版)")
            print(f"📁 所有结果已保存至: {self.config.OUTPUT_PATH}")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ 流程执行失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
        
        return self.user_df, self.cluster_profiles


if __name__ == "__main__":
    from config import Config
    
    # 初始化配置
    config = Config()
    print(config)
    
    # 初始化优化版Profiler
    profiler = UserProfiler(config)
    
    # 可选：调整优化参数
    # profiler.OPTIMIZE_NUM_CLUSTERS = False  # 关闭自动优化，使用配置的NUM_CLUSTERS
    # profiler.USE_HIERARCHICAL_CLUSTERING = True  # 启用层级聚类
    # profiler.CLUSTER_RANGE = [5, 7, 9]  # 调整候选聚类数量
    
    # 执行流程
    user_df, profiles = profiler.run()