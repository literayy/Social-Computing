import os

class Config:
    # 数据集路径配置
    BASE_PATH = r"E:\Social Computing\cresci-2017\datasets_full"
    
    # 所有数据集路径
    DATASETS = {
        'fake_followers': {
            'tweets': os.path.join(BASE_PATH, 'fake_followers.csv', 'fake_followers.csv', 'tweets.csv'),
            'users': os.path.join(BASE_PATH, 'fake_followers.csv', 'fake_followers.csv', 'users.csv'),
            'label': 'fake'
        },
        'genuine_accounts': {
            'tweets': os.path.join(BASE_PATH, 'genuine_accounts.csv', 'genuine_accounts.csv', 'tweets.csv'),
            'users': os.path.join(BASE_PATH, 'genuine_accounts.csv', 'genuine_accounts.csv', 'users.csv'),
            'label': 'genuine'
        },
        'social_spambots_1': {
            'tweets': os.path.join(BASE_PATH, 'social_spambots_1.csv', 'social_spambots_1.csv', 'tweets.csv'),
            'users': os.path.join(BASE_PATH, 'social_spambots_1.csv', 'social_spambots_1.csv', 'users.csv'),
            'label': 'fake'
        },
        'social_spambots_2': {
            'tweets': os.path.join(BASE_PATH, 'social_spambots_2.csv', 'social_spambots_2.csv', 'tweets.csv'),
            'users': os.path.join(BASE_PATH, 'social_spambots_2.csv', 'social_spambots_2.csv', 'users.csv'),
            'label': 'fake'
        },
        'social_spambots_3': {
            'tweets': os.path.join(BASE_PATH, 'social_spambots_3.csv', 'social_spambots_3.csv', 'tweets.csv'),
            'users': os.path.join(BASE_PATH, 'social_spambots_3.csv', 'social_spambots_3.csv', 'users.csv'),
            'label': 'fake'
        },
        'traditional_spambots_1': {
            'tweets': os.path.join(BASE_PATH, 'traditional_spambots_1.csv', 'traditional_spambots_1.csv', 'tweets.csv'),
            'users': os.path.join(BASE_PATH, 'traditional_spambots_1.csv', 'traditional_spambots_1.csv', 'users.csv'),
            'label': 'fake'
        },
        'traditional_spambots_2': {
            'tweets': None,
            'users': os.path.join(BASE_PATH, 'traditional_spambots_2.csv', 'traditional_spambots_2.csv', 'users.csv'),
            'label': 'fake'
        },
        'traditional_spambots_3': {
            'tweets': None,
            'users': os.path.join(BASE_PATH, 'traditional_spambots_3.csv', 'traditional_spambots_3.csv', 'users.csv'),
            'label': 'fake'
        },
        'traditional_spambots_4': {
            'tweets': None,
            'users': os.path.join(BASE_PATH, 'traditional_spambots_4.csv', 'traditional_spambots_4.csv', 'users.csv'),
            'label': 'fake'
        }
    }
    
    # 输出路径
    OUTPUT_PATH = r"E:\social_calculate\output"
    MODEL_PATH = os.path.join(OUTPUT_PATH, 'models')
    RESULT_PATH = os.path.join(OUTPUT_PATH, 'results')
    
    # 添加DeBERTa配置
    DEBERTA_MODEL_PATH = "E:\Social Computing\deberta"  # 本地DeBERTa模型文件夹路径（确保正确指向你的模型目录）
    DEBERTA_MAX_LENGTH = 512  # DeBERTa支持更长文本，建议设置为512（可根据需求调整）
    
    # XGBoost参数
    XGBOOST_PARAMS = {
        'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'random_state': 42,
            'eval_metric': 'logloss'  # 在新版本中，这个需要在初始化时指定
    }
    
    # 神经网络参数
    NN_PARAMS = {
        'hidden_layers': [256, 128, 64],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 50,
        'patience': 10
    }
    
    # 训练参数
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    @staticmethod
    def create_dirs():
        """创建必要的目录"""
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
        os.makedirs(Config.MODEL_PATH, exist_ok=True)
        os.makedirs(Config.RESULT_PATH, exist_ok=True)