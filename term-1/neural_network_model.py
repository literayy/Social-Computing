import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class FakeAccountDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class NeuralNetworkClassifier(nn.Module):
    """神经网络分类器"""
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(NeuralNetworkClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class NeuralNetworkModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        print(f"使用设备: {self.device}")
    
    def build_model(self, input_dim):
        """构建模型"""
        self.model = NeuralNetworkClassifier(
            input_dim=input_dim,
            hidden_layers=self.config.NN_PARAMS['hidden_layers'],
            dropout_rate=self.config.NN_PARAMS['dropout_rate']
        ).to(self.device)
        
        print("\n神经网络结构:")
        print(self.model)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n总参数量: {total_params:,}")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """训练模型"""
        print("\n开始训练神经网络模型...")
        
        # 构建模型
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        # 创建数据加载器
        train_dataset = FakeAccountDataset(X_train, y_train)
        val_dataset = FakeAccountDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.NN_PARAMS['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.NN_PARAMS['batch_size'],
            shuffle=False
        )
        
        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.NN_PARAMS['learning_rate']
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 早停
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # 训练循环
        for epoch in range(self.config.NN_PARAMS['epochs']):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(features).squeeze()
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(features).squeeze()
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # 调整学习率
            scheduler.step(val_loss)
            
            # 打印进度
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{self.config.NN_PARAMS['epochs']}] - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.config.NN_PARAMS['patience']:
                    print(f"\n早停触发! 在第 {epoch+1} 轮停止训练")
                    break
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print("神经网络模型训练完成!")
        
        return self.model
    
    def predict(self, X):
        """预测"""
        if self.model is None:
            raise ValueError("模型尚未训练!")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze()
            predictions = (outputs > 0.5).float().cpu().numpy()
        
        return predictions
    
    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型尚未训练!")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze().cpu().numpy()
        
        # 返回两列: [1-prob, prob]
        return np.column_stack([1 - outputs, outputs])
    
    def evaluate(self, X_test, y_test, save_path=None):
        """评估模型"""
        print("\n评估神经网络模型...")
        
        # 预测
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 打印结果
        print("\n=== 神经网络模型评估结果 ===")
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"精确率 (Precision): {metrics['precision']:.4f}")
        print(f"召回率 (Recall): {metrics['recall']:.4f}")
        print(f"F1分数 (F1-Score): {metrics['f1_score']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"\n混淆矩阵:\n{cm}")
        
        # 可视化
        if save_path:
            self._plot_confusion_matrix(cm, save_path)
            self._plot_training_history(save_path)
        
        return metrics, cm
    
    def _plot_confusion_matrix(self, cm, save_path):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Genuine', 'Fake'],
                   yticklabels=['Genuine', 'Fake'])
        plt.title('Neural Network - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{save_path}/nn_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"混淆矩阵已保存到: {save_path}/nn_confusion_matrix.png")
    
    def _plot_training_history(self, save_path):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss曲线
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy曲线
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/nn_training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练历史图已保存到: {save_path}/nn_training_history.png")
    
    def save_model(self, filepath):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型尚未训练!")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, filepath)
        print(f"神经网络模型已保存到: {filepath}")
    
    def load_model(self, filepath, input_dim):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.model is None:
            self.build_model(input_dim)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        print(f"神经网络模型已从 {filepath} 加载")
        
        return self.model