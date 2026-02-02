"""

1. 使用最佳阈值（训练时的最优值）:
   python predict.py --use-best
   或者不指定任何阈值参数（默认加载最佳阈值）:
   python predict.py

2. 使用自定义阈值:
   python predict.py --threshold 0.5
   python predict.py --threshold 0.3

3. 预测其他数据文件:
   python predict.py --data path/to/other_data.csv --threshold 0.3
   python predict.py --data Tennis-Stroke-Analysis-Data/output/test_segment.csv --use-best

4. 自定义输出文件名:
   python predict.py --output my_prediction
   # 将生成 my_prediction.csv 和 my_prediction_bounces.csv

5. 指定模型文件:
   python predict.py --model my_model.cbm --threshold 0.4

6. 完整参数示例:
   python predict.py --data Tennis-Stroke-Analysis-Data/output/training_segment.csv \
                     --model stroke_model.cbm \
                     --threshold 0.4 \
                     --output result

参数说明:
---------
--data       数据文件路径 (默认: Tennis-Stroke-Analysis-Data/output/training_segment.csv)
--model      模型文件路径 (默认: stroke_model.cbm)
--threshold  自定义预测阈值，范围 0-1 (例如: 0.4)。不指定则使用最佳阈值
--use-best   明确指定使用最佳阈值（从 best_threshold.txt 读取）
--output     输出文件前缀 (默认: predict)

输出文件:
---------
- {output_prefix}.csv: 完整预测结果（所有帧的预测概率）
- {output_prefix}_bounces.csv: 预测的击球点（pred > threshold 的点）

注意事项:
---------
1. 运行前需要先运行 train.py 进行模型训练
2. 训练时会自动生成 best_threshold.txt 文件，保存最佳阈值
3. 如果数据文件中没有 timestamp 列，将自动使用 frame_index 作为时间戳
4. 预测时会自动进行特征工程处理，与训练时保持一致
"""

import pandas as pd
import numpy as np
import os
import argparse
from catboost import CatBoostRegressor

# 导入 stroke_model.py 中的函数
from stroke_model import load_data, get_feature_cols, PREV_WINDOW_NUM, AFTER_WINDOW_NUM

# 配置
DATA_FILE = "Tennis-Stroke-Analysis-Data/output/training_segment.csv"
MODEL_PATH = "models/stroke_model.cbm"
BEST_THRESHOLD_FILE = "best_threshold.txt"


def load_best_threshold():
    """从文件中加载最佳阈值"""
    if os.path.exists(BEST_THRESHOLD_FILE):
        with open(BEST_THRESHOLD_FILE, 'r') as f:
            threshold = float(f.read().strip())
        print(f"Loaded best threshold from file: {threshold}")
        return threshold
    else:
        print(f"Warning: {BEST_THRESHOLD_FILE} not found. Using default threshold 0.4")
        return 0.4


def predict_with_model(test_data, model, threshold, output_prefix="predict"):
    """使用模型进行预测并保存结果"""
    print(f"Running prediction on data (size: {len(test_data)})...")
    
    # 进行预测
    test_data["pred"] = model.predict(test_data[get_feature_cols(PREV_WINDOW_NUM, AFTER_WINDOW_NUM)])
    
    # 检查是否有 timestamp 列，如果没有则创建
    if 'timestamp' not in test_data.columns:
        test_data['timestamp'] = test_data['frame_index']
        print("Note: 'timestamp' column not found, using 'frame_index' as timestamp")
    
    # 选择要保存的列
    output_cols = ["frame_index", "timestamp", "pred", "event_cls", "x", "y"]
    
    # 确保所有列都存在
    available_cols = [col for col in output_cols if col in test_data.columns]
    
    # 保存完整预测结果
    predict_file = f"{output_prefix}.csv"
    test_data[available_cols].to_csv(predict_file, index=False, encoding='utf-8')
    print(f"Saved full predictions to {predict_file}")
    
    # 保存预测的击球点数据（pred > threshold的点）
    predicted_bounces = test_data[test_data["pred"] > threshold][available_cols]
    bounces_file = f"{output_prefix}_bounces.csv"
    predicted_bounces.to_csv(bounces_file, index=False, encoding='utf-8')
    print(f"Saved {len(predicted_bounces)} predicted stroke points to {bounces_file} (threshold={threshold:.4f})")
    
    # 打印统计信息
    if 'event_cls' in test_data.columns:
        actual_positives = test_data['event_cls'].sum()
        predicted_positives = len(predicted_bounces)
        print(f"\nStatistics:")
        print(f"  Actual positive samples: {actual_positives}")
        print(f"  Predicted positive samples: {predicted_positives}")
        print(f"  Prediction rate: {predicted_positives / len(test_data) * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Tennis Stroke Prediction Script')
    parser.add_argument('--data', type=str, default=DATA_FILE, 
                        help=f'Path to data file (default: {DATA_FILE})')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help=f'Path to model file (default: {MODEL_PATH})')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Custom threshold for prediction (e.g., 0.4). If not specified, uses best threshold from training.')
    parser.add_argument('--use-best', action='store_true',
                        help='Use best threshold from training (stored in best_threshold.txt)')
    parser.add_argument('--output', type=str, default='predict',
                        help='Output file prefix (default: predict, will generate predict.csv and predict_bounces.csv)')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件 {args.model} 不存在。请先运行 stroke_model.py 进行训练。")
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"数据文件 {args.data} 不存在。")
    
    # 确定使用的阈值
    if args.threshold is not None:
        # 使用自定义阈值
        threshold = args.threshold
        print(f"Using custom threshold: {threshold}")
    elif args.use_best or args.threshold is None:
        # 使用最佳阈值（从文件加载）
        threshold = load_best_threshold()
    else:
        # 默认阈值
        threshold = 0.4
        print(f"Using default threshold: {threshold}")
    
    # 加载模型
    print(f"Loading model from {args.model}...")
    catboost_regressor = CatBoostRegressor()
    catboost_regressor.load_model(args.model)
    
    # 加载并处理数据（生成特征）
    print(f"Loading and processing data from {args.data}...")
    all_data = load_data(args.data, shuffle=False)  # shuffle=False 保持时序
    
    print(f"Data loaded: {len(all_data)} samples, {all_data['event_cls'].sum()} positive samples")
    
    # 进行预测
    predict_with_model(all_data, catboost_regressor, threshold, output_prefix=args.output)
    
    print("\nPrediction completed successfully!")


if __name__ == "__main__":
    main()
