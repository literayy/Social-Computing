# 虚假账号识别实验

基于Cresci-2017数据集的Twitter虚假账号识别系统，结合传统机器学习特征和深度学习BERT特征，使用XGBoost和神经网络进行对比实验。

## 项目结构

```
social_calculate/
├── config.py                    # 配置文件
├── data_preprocessing.py        # 数据预处理模块
├── feature_extraction.py        # 特征提取模块
├── xgboost_model.py            # XGBoost模型
├── neural_network_model.py     # 神经网络模型
├── main.py                      # 主程序
├── requirements.txt             # 依赖包
├── README.md                    # 说明文档
├── cresci-2017/                 # 数据集目录
│   └── datasets_full/
│       ├── fake_followers.csv/
│       ├── genuine_accounts.csv/
│       ├── social_spambots_1.csv/
│       ├── social_spambots_2.csv/
│       ├── social_spambots_3.csv/
│       ├── traditional_spambots_1.csv/
│       ├── traditional_spambots_2.csv/
│       ├── traditional_spambots_3.csv/
│       └── traditional_spambots_4.csv/
└── output/                      # 输出目录
    ├── models/                  # 模型保存
    ├── results/                 # 结果保存
    └── preprocessed_data.csv    # 预处理后的数据
```

## 功能特点

### 1. 数据预处理
- 加载所有9个Cresci-2017子数据集
- 数据清洗（去重、缺失值处理）
- 聚合用户的推文特征
- 标签编码（genuine=0, fake=1）

### 2. 特征提取

#### 传统特征（50维）
- **账户基本特征**（10维）：粉丝数、关注数、推文数、点赞数、列表数、粉丝关注比、推文频率、账户年龄、认证状态等
- **用户资料特征**（10维）：用户名长度、描述长度、是否包含数字、URL数量、相似度等
- **推文内容特征**（15维）：推文长度、标签数、提及数、URL数量、转发数、点赞数统计等
- **行为特征**（10维）：转发率、点赞率、互动率、活跃度等
- **时序特征**（5维）：账号创建时间、是否新账号等

#### BERT特征（768维）
- 使用bert-base-uncased模型
- 提取推文文本的[CLS] token表示
- 捕获语义信息

#### 特征融合
- 传统特征标准化
- 与BERT特征拼接
- 最终特征维度：818维

### 3. 模型

#### XGBoost模型
- 梯度提升树算法
- 自动特征重要性分析
- 高效的并行训练
- 超参数可配置

#### 神经网络模型
- 多层全连接网络
- Batch Normalization
- Dropout正则化
- 学习率自适应调整
- 早停机制

### 4. 评估指标
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数（F1-Score）
- AUC值

### 5. 可视化
- 混淆矩阵
- 特征重要性图（XGBoost）
- 训练历史曲线（神经网络）
- 模型性能对比图
- 雷达图对比

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- pandas：数据处理
- numpy：数值计算
- scikit-learn：机器学习工具
- xgboost：XGBoost模型
- torch：PyTorch深度学习框架
- transformers：BERT模型
- matplotlib/seaborn：可视化

## 使用方法

### 1. 配置路径

在 `config.py` 中修改数据集路径：

```python
BASE_PATH = r"E:\social_calculate\cresci-2017\datasets_full"
OUTPUT_PATH = r"E:\social_calculate\output"
```

### 2. 运行实验

```bash
python main.py
```

### 3. 查看结果

实验完成后，结果保存在 `output/` 目录：

- `models/`：训练好的模型文件
  - `xgboost_model.pkl`
  - `neural_network_model.pth`
- `results/`：评估结果和可视化
  - `results_summary.txt`：文本格式结果
  - `results_detailed.csv`：CSV格式结果
  - `xgboost_confusion_matrix.png`
  - `xgboost_feature_importance.png`
  - `nn_confusion_matrix.png`
  - `nn_training_history.png`
  - `model_comparison.png`
- `preprocessed_data.csv`：预处理后的数据
- `combined_features.npy`：合并特征
- `traditional_features.npy`：传统特征
- `labels.npy`：标签

## 实验流程

```
1. 数据预处理
   ├── 加载9个数据集
   ├── 数据清洗
   └── 特征聚合

2. 特征提取
   ├── 提取50维传统特征
   ├── 提取768维BERT特征
   └── 特征融合（818维）

3. 数据划分
   ├── 训练集：64%
   ├── 验证集：16%
   └── 测试集：20%

4. XGBoost训练
   ├── 模型训练
   ├── 验证集评估
   └── 测试集评估

5. 神经网络训练
   ├── 模型训练
   ├── 验证集评估
   └── 测试集评估

6. 结果对比
   ├── 生成对比报告
   ├── 绘制可视化图表
   └── 保存所有结果
```

## 参数配置

### XGBoost参数
```python
XGBOOST_PARAMS = {
    'max_depth': 6,              # 树的最大深度
    'learning_rate': 0.1,        # 学习率
    'n_estimators': 200,         # 树的数量
    'objective': 'binary:logistic',
    'random_state': 42,
    'n_jobs': -1
}
```

### 神经网络参数
```python
NN_PARAMS = {
    'hidden_layers': [256, 128, 64],  # 隐藏层
    'dropout_rate': 0.3,              # Dropout率
    'learning_rate': 0.001,           # 学习率
    'batch_size': 64,                 # 批次大小
    'epochs': 50,                     # 训练轮数
    'patience': 10                    # 早停耐心值
}
```

### BERT参数
```python
BERT_MODEL = 'bert-base-uncased'
BERT_MAX_LENGTH = 128
```

## 数据集说明

Cresci-2017数据集包含9个子集：

| 数据集 | 类型 | 说明 |
|--------|------|------|
| genuine_accounts | 真实 | 真实用户账号 |
| fake_followers | 虚假 | 购买的虚假粉丝 |
| social_spambots_1 | 虚假 | 社交垃圾机器人1 |
| social_spambots_2 | 虚假 | 社交垃圾机器人2 |
| social_spambots_3 | 虚假 | 社交垃圾机器人3 |
| traditional_spambots_1 | 虚假 | 传统垃圾机器人1 |
| traditional_spambots_2 | 虚假 | 传统垃圾机器人2 |
| traditional_spambots_3 | 虚假 | 传统垃圾机器人3 |
| traditional_spambots_4 | 虚假 | 传统垃圾机器人4 |

每个数据集包含：
- `users.csv`：用户资料数据
- `tweets.csv`：推文数据（部分数据集包含）

## 常见问题

### 1. 内存不足
如果遇到内存问题，可以：
- 减少BERT的batch_size
- 使用更小的BERT模型（如bert-base）
- 减少训练样本数量

### 2. CUDA不可用
如果没有GPU，程序会自动使用CPU，但速度较慢。建议：
- 减少BERT处理的文本数量
- 使用预训练好的特征
- 仅使用传统特征

### 3. 文件路径问题
确保路径使用原始字符串（r"..."）或正斜杠（/）

## 性能预期

在完整的Cresci-2017数据集上：

- **XGBoost**：
  - 准确率：95%+
  - F1分数：94%+
  - 训练时间：5-10分钟

- **神经网络**：
  - 准确率：93%+
  - F1分数：92%+
  - 训练时间：10-20分钟

## 扩展建议

1. **特征工程**：
   - 添加更多时序特征
   - 网络结构特征
   - 文本情感分析

2. **模型优化**：
   - 超参数网格搜索
   - 集成学习方法
   - 尝试其他深度学习模型

3. **数据增强**：
   - 处理不平衡数据
   - 交叉验证
   - 数据增强技术

## 参考文献

- Cresci, S., Di Pietro, R., Petrocchi, M., Spognardi, A., & Tesconi, M. (2017). The paradigm-shift of social spambots: Evidence, theories, and tools for the arms race. In Proceedings of the 26th international conference on world wide web companion (pp. 963-972).

## 许可证

MIT License

## 作者

虚假账号识别实验项目

## 更新日志

- v1.0 (2024): 初始版本
  - 完整的数据预处理流程
  - 50维传统特征提取
  - BERT特征提取
  - XGBoost和神经网络模型
  - 完整的评估和可视化