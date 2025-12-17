import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块（移除DebertaFeatureExtractor）
from config import Config
from data_preprocessing import DataPreprocessor
from feature_extraction import FeatureExtractor
from xgboost_model import XGBoostModel
from neural_network_model import NeuralNetworkModel

# ==================== 配置开关 ====================
SKIP_PREPROCESSING = True  # 设置为True跳过数据预处理
SKIP_FEATURE_EXTRACTION = True  # 设置为True跳过特征提取
# =================================================

def save_results_to_file(results, filepath):
    """保存结果到文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("虚假账号识别实验结果（仅传统特征）\n")
        f.write("=" * 60 + "\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"\n{model_name} 模型结果:\n")
            f.write("-" * 40 + "\n")
            f.write(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}\n")
            f.write(f"精确率 (Precision): {metrics['precision']:.4f}\n")
            f.write(f"召回率 (Recall):    {metrics['recall']:.4f}\n")
            f.write(f"F1分数 (F1-Score):  {metrics['f1_score']:.4f}\n")
            f.write(f"AUC:               {metrics['auc']:.4f}\n")
            f.write("\n")
    
    print(f"\n结果已保存到: {filepath}")

def plot_model_comparison(results, save_path):
    """绘制模型对比图"""
    models = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # 为每个指标绘制柱状图
    for idx, metric in enumerate(metrics_names):
        values = [results[model][metric] for model in models]
        axes[idx].bar(models, values, color=['skyblue', 'lightcoral'])
        axes[idx].set_ylabel(metric.upper())
        axes[idx].set_title(f'{metric.upper()} Comparison (仅传统特征)')
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # 在柱子上显示数值
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
    
    # 综合对比雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(2, 3, 6, projection='polar')
    
    for model in models:
        values = [results[model][metric] for metric in metrics_names]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics_names])
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance Comparison (仅传统特征)', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/model_comparison_traditional_only.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"模型对比图已保存到: {save_path}/model_comparison_traditional_only.png")

def load_existing_features(config):
    """加载已保存的传统特征和标签"""
    print("\n" + "=" * 60)
    print("加载已保存的传统特征文件...")
    print("=" * 60)
    
    traditional_features_path = f"{config.OUTPUT_PATH}/traditional_features_scaled.npy"
    labels_path = f"{config.OUTPUT_PATH}/labels.npy"
    
    # 检查文件是否存在
    if not os.path.exists(traditional_features_path):
        raise FileNotFoundError(f"未找到传统特征文件: {traditional_features_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"未找到标签文件: {labels_path}")
    
    # 加载特征和标签
    traditional_scaled = np.load(traditional_features_path)
    y = np.load(labels_path)
    
    print(f"✓ 成功加载传统特征文件")
    print(f"  特征形状: {traditional_scaled.shape}")
    print(f"  标签形状: {y.shape}")
    print(f"  标签分布: {np.bincount(y.astype(int))}")
    
    return traditional_scaled, y

def main():
    """主函数"""
    print("=" * 60)
    print("虚假账号识别实验（仅传统特征）")
    print("=" * 60)
    
    # 创建配置
    config = Config()
    config.create_dirs()
    
    # ==================== 第一步: 数据预处理（可选） ====================
    if not SKIP_PREPROCESSING:
        print("\n" + "=" * 60)
        print("第一步: 数据预处理")
        print("=" * 60)
        
        preprocessor = DataPreprocessor(config)
        df = preprocessor.preprocess()
        
        # 检查数据
        if df.empty or 'label' not in df.columns:
            print("错误: 数据为空或缺少标签列!")
            return
        
        print(f"\n数据集统计:")
        print(f"总样本数: {len(df)}")
        print(f"特征数: {len(df.columns) - 1}")
        print(f"标签分布:\n{df['label'].value_counts()}")
    else:
        print("\n⏭️  跳过数据预处理步骤")
    
    # ==================== 第二步: 特征提取（可选） ====================
    if not SKIP_FEATURE_EXTRACTION:
        print("\n" + "=" * 60)
        print("第二步: 特征提取（仅传统特征）")
        print("=" * 60)
        
        # 如果跳过了预处理，需要加载预处理后的数据
        if SKIP_PREPROCESSING:
            preprocessed_path = f"{config.OUTPUT_PATH}/preprocessed_data.csv"
            if not os.path.exists(preprocessed_path):
                raise FileNotFoundError(f"未找到预处理数据: {preprocessed_path}")
            df = pd.read_csv(preprocessed_path)
            print(f"✓ 加载预处理数据: {len(df)} 条记录")
        
        feature_extractor = FeatureExtractor(config)
        
        # 仅提取传统特征
        traditional_features = feature_extractor.extract_traditional_features(df)
        
        # 标准化传统特征
        traditional_scaled = feature_extractor.scaler.fit_transform(traditional_features)
        print(f"传统特征维度: {traditional_scaled.shape}")
        
        # 准备标签
        y = df['label'].values
        
        # 保存传统特征
        np.save(f"{config.OUTPUT_PATH}/traditional_features_scaled.npy", traditional_scaled)
        np.save(f"{config.OUTPUT_PATH}/labels.npy", y)
        print(f"\n传统特征已保存到: {config.OUTPUT_PATH}")
    else:
        print("\n⏭️  跳过特征提取步骤")
        # 加载已保存的传统特征
        traditional_scaled, y = load_existing_features(config)
    
    # ==================== 第三步: 数据划分 ====================
    print("\n" + "=" * 60)
    print("第三步: 数据划分")
    print("=" * 60)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        traditional_scaled, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    # 从训练集中划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=y_train
    )
    
    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    print(f"特征维度: {X_train.shape[1]}")
    
    # ==================== 第四步: XGBoost模型训练 ====================
    print("\n" + "=" * 60)
    print("第四步: XGBoost模型训练与评估")
    print("=" * 60)
    
    xgb_model = XGBoostModel(config)
    xgb_model.train(X_train, y_train, X_val, y_val)
    xgb_metrics, xgb_cm = xgb_model.evaluate(
        X_test, y_test, 
        save_path=config.RESULT_PATH
    )
    
    # ==================== 第五步: 神经网络模型训练 ====================
    print("\n" + "=" * 60)
    print("第五步: 神经网络模型训练与评估")
    print("=" * 60)
    
    nn_model = NeuralNetworkModel(config)
    nn_model.train(X_train, y_train, X_val, y_val)
    nn_metrics, nn_cm = nn_model.evaluate(
        X_test, y_test,
        save_path=config.RESULT_PATH
    )
    
    # ==================== 第六步: 结果对比 ====================
    print("\n" + "=" * 60)
    print("第六步: 模型对比与结果保存")
    print("=" * 60)
    
    results = {
        'XGBoost (传统特征)': xgb_metrics,
        'Neural Network (传统特征)': nn_metrics
    }
    
    # 打印对比结果
    print("\n" + "=" * 60)
    print("模型性能对比（仅传统特征）")
    print("=" * 60)
    print(f"\n{'指标':<15} {'XGBoost':<15} {'Neural Network':<15}")
    print("-" * 45)
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
        xgb_val = results['XGBoost (传统特征)'][metric]
        nn_val = results['Neural Network (传统特征)'][metric]
        print(f"{metric.upper():<15} {xgb_val:<15.4f} {nn_val:<15.4f}")
    
    # 保存结果到文件
    save_results_to_file(results, f"{config.RESULT_PATH}/results_summary_traditional_only.txt")
    
    # 绘制模型对比图
    plot_model_comparison(results, config.RESULT_PATH)
    
    # 保存详细结果到CSV
    results_df = pd.DataFrame({
        'Model': ['XGBoost (传统特征)', 'Neural Network (传统特征)'],
        'Accuracy': [xgb_metrics['accuracy'], nn_metrics['accuracy']],
        'Precision': [xgb_metrics['precision'], nn_metrics['precision']],
        'Recall': [xgb_metrics['recall'], nn_metrics['recall']],
        'F1-Score': [xgb_metrics['f1_score'], nn_metrics['f1_score']],
        'AUC': [xgb_metrics['auc'], nn_metrics['auc']]
    })
    results_df.to_csv(f"{config.RESULT_PATH}/results_detailed_traditional_only.csv", index=False)
    print(f"\n详细结果已保存到: {config.RESULT_PATH}/results_detailed_traditional_only.csv")
    
    print("\n" + "=" * 60)
    print("实验完成!（仅传统特征）")
    print("=" * 60)
    print(f"\n所有结果已保存到: {config.OUTPUT_PATH}")
    print(f"  - 结果文件: {config.RESULT_PATH}")
    print(f"  - 核心对比图: {config.RESULT_PATH}/model_comparison_traditional_only.png")

if __name__ == "__main__":
    main()