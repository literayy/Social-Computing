import os
import torch
import numpy as np

class Config:
    """配置文件 - 完整版 (步骤1-6)"""
    
    def __init__(self):
        # ==================== 路径配置 ====================
        self.BASE_PATH = r"F:\Social Computing\term2"
        self.BASE_PATH_data = r"E:\Social Computing"
        
        # 数据路径
        self.DATA_PATH = os.path.join(self.BASE_PATH_data, "data")
        self.CRESCI_PATH = os.path.join(self.DATA_PATH, "cresci-2017")
        self.GENDER_PATH = os.path.join(self.DATA_PATH, "gender-classifier-DFE-791531.csv")
        
        # 预处理数据路径
        self.PREPROCESSED_CRESCI = os.path.join(self.BASE_PATH, "output", "preprocessed_cresci.csv")
        self.PREPROCESSED_GENDER = os.path.join(self.BASE_PATH, "output", "preprocessed_gender.csv")
        self.PREPROCESSED_COMBINED = os.path.join(self.BASE_PATH, "output", "preprocessed_combined_data.csv")
        
        # 模型路径
        self.DEBERTA_PATH = os.path.join(self.BASE_PATH_data, "deberta")
        
        # 输出路径
        self.OUTPUT_PATH = os.path.join(self.BASE_PATH, "output")
        self.MODEL_SAVE_PATH = os.path.join(self.OUTPUT_PATH, "models")
        self.LOG_PATH = os.path.join(self.OUTPUT_PATH, "logs")
        os.makedirs(self.OUTPUT_PATH, exist_ok=True)
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(self.LOG_PATH, exist_ok=True)
        
        # ==================== 数据预处理配置 (步骤1) ====================
        self.MIN_TEXT_LENGTH = 10
        self.RANDOM_SEED = 42
        self.CRESCI_SAMPLE_SIZE = 5000
        
        # ==================== 动态采样配置 (步骤2) ====================
        self.USE_DYNAMIC_SAMPLING = True
        self.DYNAMIC_SAMPLE_SIZE = 500
        self.SAMPLING_EPOCH = 1
        self.BALANCED_SAMPLING = True
        
        # ==================== 设备检测 ====================
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # ==================== 模型训练配置 (步骤2) ====================
        if self.device_type == 'cpu':
            self.MAX_LENGTH = 64
            self.BATCH_SIZE = 4
            self.NUM_EPOCHS = 20
            self.MAX_TRAIN_SAMPLES = 500
            self.MAX_VAL_SAMPLES = 100
            self.MAX_TEST_SAMPLES = 500
            self.FREEZE_LAYERS = 10
            self.LEARNING_RATE = 2e-5
            self.WEIGHT_DECAY = 0.01
            print("\n⚠️  检测到CPU模式 - 使用优化配置")
        else:
            self.MAX_LENGTH = 128
            self.BATCH_SIZE = 16
            self.NUM_EPOCHS = 10
            self.MAX_TRAIN_SAMPLES = None
            self.MAX_VAL_SAMPLES = None
            self.MAX_TEST_SAMPLES = None
            self.FREEZE_LAYERS = 0
            self.LEARNING_RATE = 2e-5
            self.WEIGHT_DECAY = 0.001
            print("\n✓ 检测到GPU模式")
        
        self.DEBERTA_HIDDEN_SIZE = 1024
        self.WARMUP_RATIO = 0.1
        self.MAX_GRAD_NORM = 1.0
        self.GRADIENT_ACCUMULATION_STEPS = 1
        
        # 域适应参数
        self.LAMBDA_DOMAIN = 0.1
        self.LAMBDA_TASK = 1.0
        self.GRL_ALPHA = 1.0
        
        # 任务配置
        self.NUM_BOT_CLASSES = 2
        self.NUM_GENDER_CLASSES = 3
        self.NUM_DOMAINS = 2
        
        # 数据集划分
        self.TRAIN_RATIO = 0.7
        self.VAL_RATIO = 0.15
        self.TEST_RATIO = 0.15
        
        # ==================== 用户画像配置 (步骤3) - 新增 ====================
        self.NUM_CLUSTERS = 8  # 聚类数量
        self.PROFILING_SAMPLE_SIZE = 3000  # 画像构建采样数量
        
        # ==================== 冷启动配置 (步骤4) - 新增 ====================
        self.COLD_START_TEST_SIZES = [1, 3, 5, 10]  # 测试的推文数量
        self.KNN_NEIGHBORS = 3  # KNN检索邻居数
        
        # ==================== 鲁棒性测试配置 (步骤5) - 新增 ====================
        self.ROBUSTNESS_TEST_SIZE = 500  # 鲁棒性测试样本数
        self.ATTACK_TYPES = ['word_swap', 'char_insert']  # 对抗攻击类型
        
        # ==================== 显示配置 ====================
        self.VERBOSE = True
        self.LOG_INTERVAL = 10 if self.device_type == 'cpu' else 50
        self.SAVE_INTERVAL = 1
        self.SAVE_LATEST_CHECKPOINT = True
        
    def __repr__(self):
        config_str = "\n" + "=" * 60 + "\n"
        config_str += "配置信息 (步骤1-6: 完整版)\n"
        config_str += "=" * 60 + "\n"
        config_str += f"运行模式: {self.device_type.upper()}\n"
        config_str += f"Cluster数量: {self.NUM_CLUSTERS}\n"
        config_str += f"冷启动测试: {self.COLD_START_TEST_SIZES}\n"
        config_str += "-" * 60 + "\n"
        for key, value in self.__dict__.items():
            if not key.startswith('_') and not isinstance(value, (np.ndarray, torch.Tensor)):
                config_str += f"{key}: {value}\n"
        config_str += "=" * 60
        return config_str


if __name__ == "__main__":
    config = Config()
    print(config)
    
    print(f"\n✓ 输出路径: {config.OUTPUT_PATH}")
    print(f"✓ 模型保存路径: {config.MODEL_SAVE_PATH}")
    print(f"✓ 日志路径: {config.LOG_PATH}")