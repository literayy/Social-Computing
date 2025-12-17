"""
XGBoost模型训练和评估模块（修复版）
"""
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class XGBoostModel:
    def __init__(self, config):
        """初始化XGBoost模型"""
        self.config = config
        # 将 eval_metric 从参数中分离出来
        self.params = config.XGBOOST_PARAMS.copy()
        self.eval_metric = self.params.pop('eval_metric', 'logloss')
        
        # 在初始化时就指定 eval_metric
        self.model = xgb.XGBClassifier(
            eval_metric=self.eval_metric,
            **self.params
        )
    
    def train(self, X_train, y_train, X_val, y_val):
        """训练XGBoost模型"""
        print("开始训练XGBoost模型...")
        
        # 不再在 fit() 中传递 eval_metric
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        
        print("✓ XGBoost模型训练完成")
        
        return self.model
    
    def evaluate(self, X_test, y_test, save_path=None):
        """评估模型性能，支持保存混淆矩阵"""
        print("\n评估XGBoost模型...")
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 打印结果
        print("\nXGBoost模型性能:")
        print(f"  准确率 (Accuracy):  {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision): {metrics['precision']:.4f}")
        print(f"  召回率 (Recall):    {metrics['recall']:.4f}")
        print(f"  F1分数 (F1-Score):  {metrics['f1_score']:.4f}")
        print(f"  AUC:               {metrics['auc']:.4f}")
        
        # 保存混淆矩阵（如果提供了保存路径）
        if save_path:
            # 确保保存目录存在
            os.makedirs(save_path, exist_ok=True)
            # 绘制混淆矩阵
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.title('XGBoost模型混淆矩阵')
            plt.tight_layout()
            # 保存图像
            cm_path = os.path.join(save_path, 'xgboost_confusion_matrix.png')
            plt.savefig(cm_path, dpi=300)
            plt.close()
            print(f"✓ 混淆矩阵已保存至: {cm_path}")
        
        return metrics, cm
    
    def plot_feature_importance(self, feature_names=None, top_n=20):
        """绘制特征重要性"""
        print("\n绘制特征重要性...")
        
        importance = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        # 排序并选择前N个特征
        indices = importance.argsort()[-top_n:][::-1]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importance[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('特征重要性')
        plt.title(f'XGBoost 前{top_n}个重要特征')
        plt.tight_layout()
        
        # 修复路径引用（使用config中定义的OUTPUT_PATH）
        save_path = os.path.join(self.config.OUTPUT_PATH, 'xgboost_feature_importance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 特征重要性图已保存: {save_path}")
        plt.close()
    
    def save_model(self, filename='xgboost_model.pkl'):
        """保存模型"""
        # 修复路径引用
        save_path = os.path.join(self.config.MODEL_PATH, filename)
        joblib.dump(self.model, save_path)
        print(f"✓ XGBoost模型已保存: {save_path}")
    
    def load_model(self, filename='xgboost_model.pkl'):
        """加载模型"""
        # 修复路径引用
        load_path = os.path.join(self.config.MODEL_PATH, filename)
        self.model = joblib.load(load_path)
        print(f"✓ XGBoost模型已加载: {load_path}")