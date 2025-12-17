import pandas as pd
import numpy as np
import re
import torch
import gc
from transformers import DebertaV2Tokenizer, DebertaV2Model
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class DebertaFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型和tokenizer
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
    
    def _load_tokenizer(self):
        """加载DeBERTa tokenizer"""
        try:
            tokenizer = DebertaV2Tokenizer.from_pretrained("./deberta")
            print("✓ Tokenizer加载成功")
            return tokenizer
        except Exception as e:
            raise Exception(f"Tokenizer加载失败: {str(e)}")
    
    def _load_model(self):
        """加载DeBERTa模型"""
        try:
            model = DebertaV2Model.from_pretrained("./deberta").to(self.device)
            model.eval()
            print("✓ 模型加载成功并设置为评估模式")
            return model
        except Exception as e:
            raise Exception(f"模型加载失败: {str(e)}")
    
    def extract_deberta_features_from_chunk(self, data_path, text_column='all_tweets_text'):
        """开始提取DeBERTa特征"""
        print(f"\n开始提取DeBERTa特征...")
        print(f"数据路径: {data_path}")
        print(f"文本列: {text_column}")
        
        # 读取预处理数据
        df = self._load_data(data_path)
        texts = self._prepare_texts(df, text_column)
        
        # 配置参数
        batch_size = 8
        chunk_size = 1000
        start_chunk_idx = 0 
        start_index = start_chunk_idx * chunk_size  # 起始索引位置
        
        # 检查起始索引是否合法
        if start_index >= len(texts):
            print(f"警告: 起始索引 {start_index} 超出数据长度 {len(texts)}")
            print("没有更多数据需要处理")
            return
        
        # 创建保存目录
        save_dir = os.path.join(self.config.OUTPUT_PATH, 'deberta_chunks')
        os.makedirs(save_dir, exist_ok=True)
        print(f"特征保存目录: {save_dir}")
        
        # 开始处理
        with torch.no_grad():
            for chunk_idx in range(start_chunk_idx, len(texts) // chunk_size + 1):
                current_start = chunk_idx * chunk_size
                current_end = min((chunk_idx + 1) * chunk_size, len(texts))
                
                if current_start >= len(texts):
                    break
                
                chunk_texts = texts[current_start:current_end]
                print(f"\n处理chunk {chunk_idx} (索引范围: {current_start}-{current_end})")
                print(f"当前chunk文本数量: {len(chunk_texts)}")
                
                chunk_features = []
                
                # 批量处理
                for i in range(0, len(chunk_texts), batch_size):
                    mini_batch = chunk_texts[i:i+batch_size]
                    
                    # Tokenize
                    encoded = self.tokenizer(
                        mini_batch,
                        padding=True,
                        truncation=True,
                        max_length=self.config.DEBERTA_MAX_LENGTH,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    # 获取特征
                    outputs = self.model(**encoded)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    chunk_features.extend(cls_embeddings)
                    
                    # 释放内存
                    del encoded, outputs, cls_embeddings
                    torch.cuda.empty_cache() if self.device.type == 'cuda' else None
                    gc.collect()
                    
                    # 打印进度
                    if (i // batch_size + 1) % 5 == 0:
                        processed = min(i + batch_size, len(chunk_texts))
                        print(f"  批次进度: {processed}/{len(chunk_texts)}")
                
                # 保存当前chunk
                chunk_save_path = os.path.join(save_dir, f'deberta_features_chunk_{chunk_idx}.npy')
                np.save(chunk_save_path, chunk_features)
                print(f"✓ 已保存chunk {chunk_idx} 到 {chunk_save_path}")
                print(f"  特征形状: {np.array(chunk_features).shape}")
                
                # 释放chunk内存
                del chunk_features
                gc.collect()
        
        print("\n✅ 所有chunk处理完成！")
    
    def _load_data(self, data_path):
        """加载预处理数据"""
        try:
            df = pd.read_csv(data_path)
            print(f"✓ 成功读取数据，共 {len(df)} 条记录")
            return df
        except Exception as e:
            raise Exception(f"数据读取失败: {str(e)}")
    
    def _prepare_texts(self, df, text_column):
        """准备文本数据"""
        if text_column in df.columns:
            texts = df[text_column].fillna('').astype(str).tolist()
        else:
            print(f"警告: 未找到 {text_column} 列，使用 description 列替代")
            texts = df['description'].fillna('').astype(str).tolist()
        
        print(f"✓ 准备完成 {len(texts)} 条文本")
        return texts

# 配置类
class Config:
    def __init__(self):
        self.OUTPUT_PATH = r"E:\social_calculate\output"  # 输出目录（与你的路径一致）
        self.DEBERTA_MAX_LENGTH = 512  # 根据模型要求调整

if __name__ == "__main__":
    # 初始化配置
    config = Config()

    # 初始化特征提取器
    extractor = DebertaFeatureExtractor(config)
    
    # 数据路径（你的预处理数据）
    data_path = r"E:\social_calculate\output\preprocessed_data.csv"
    
    try:
        extractor.extract_deberta_features_from_chunk(data_path)
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {str(e)}")
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()