import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt

from stroke_model import load_data, get_feature_cols, PREV_WINDOW_NUM, AFTER_WINDOW_NUM


DATA_FILE = "Tennis-Stroke-Analysis-Data/output/training_segment.csv"

def train(train_data, test_data):
    if train_data["event_cls"].nunique() < 2:  # 统计event_cls列中唯一值的数量，如果小于2，则说明只有单一类别，即没有正样本
        raise ValueError("训练集中只有单一类别（event_cls 全为同一值）。请检查 bounce_train.json 是否包含正样本，或重新生成标注数据。")
    
    catboost_regressor = CatBoostRegressor(iterations=3000, depth=3, learning_rate=0.1, loss_function='RMSE')
    catboost_regressor.fit(
        train_data[get_feature_cols(PREV_WINDOW_NUM, AFTER_WINDOW_NUM)],  # 训练特征
        train_data['event_cls'],                                         # 训练标签
        eval_set=(test_data[get_feature_cols(PREV_WINDOW_NUM, AFTER_WINDOW_NUM)], test_data['event_cls']),  # 验证集
        use_best_model=True,                                             # 使用最佳模型
        sample_weight=train_data['weight'],                              # 样本权重
        early_stopping_rounds=100,                                    # 早停轮数（注释掉了）
    )
    return catboost_regressor


def evaluate(train_data, test_data, catboost_regressor):
    test_data["pred"] = catboost_regressor.predict(test_data[get_feature_cols(PREV_WINDOW_NUM, AFTER_WINDOW_NUM)])
    
    # 存储每个阈值的指标
    thresholds = []
    accuracies = []
    recalls = []
    precisions = []
    f1_scores = []
    f_beta_scores = []  # 新增F-beta分数
    
    for threshold in np.arange(0.2, 0.50, 0.01):
        # print(f'===> threshold: {threshold}')

        # 使用 sklearn 计算混淆矩阵
        pred_labels = (test_data["pred"] > threshold).astype(int)
        cm = confusion_matrix(test_data['event_cls'], pred_labels)
        tn, fp, fn, tp = cm.ravel()  # [[tn, fp], [fn, tp]]
        
        print(f'tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}, total: {tn + tp + fn + fp}')

        acc = (tn + tp) / (tn + tp + fn + fp) if (tn + tp + fn + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = 0.3 + tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算F-beta分数 (beta^2 = 4, 所以召回率权重是精确率的4倍)
        beta_squared = 8
        f_beta = (1 + beta_squared) * precision * recall / (beta_squared * precision + recall) if (beta_squared * precision + recall) > 0 else 0
        
        print(f'threshold: {threshold:.3f}, accuracy: {acc:.3f}, recall: {recall:.3f}, precision: {precision:.3f}, f1: {f1:.3f}, f-beta: {f_beta:.3f}')
        
        thresholds.append(threshold)
        accuracies.append(acc)
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
        f_beta_scores.append(f_beta)

    # 选择最佳阈值（最大化F-beta分数，更重视召回率）
    best_idx = np.argmax(f_beta_scores)
    best_threshold = thresholds[best_idx]
    print(f'Best threshold: {best_threshold} with F-beta: {f_beta_scores[best_idx]} (F1: {f1_scores[best_idx]})')

    print("roc", roc_auc_score(test_data['event_cls'], test_data['pred']))
    
    # 计算AUC-PR
    precision_curve, recall_curve, _ = precision_recall_curve(test_data['event_cls'], test_data['pred'])
    auc_pr = auc(recall_curve, precision_curve)
    print(f'AUC-PR: {auc_pr}')
    
    # # 绘制PR曲线并保存
    # plt.figure(figsize=(8, 6))
    # plt.plot(recall_curve, precision_curve, label=f'PR Curve (AUC = {auc_pr:.3f})')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('pr_curve.png')
    # plt.close()
    # print("PR curve saved as pr_curve.png")
    
    # 保存最佳阈值到文件，供 predict.py 使用
    with open('best_threshold.txt', 'w') as f:
        f.write(str(best_threshold))
    print(f"Best threshold saved to best_threshold.txt")
    
    return best_threshold


def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: Dataset {DATA_FILE} not found.")
        return None

    # 1. 加载所有数据
    # 注意：load_data 内部现在默认是 shuffle=True，但为了分割数据集，我们需要控制它
    all_data = load_data(DATA_FILE, shuffle=False)
    
    print(f"Total data shape: {all_data.shape}, positive samples: {len(all_data[all_data['event_cls'] == 1])}")
    
    # 2. 按轨迹(traj_id)进行训练/测试分割 (Group Split)
    # 防止同一条轨迹的数据一部分在训练集，一部分在测试集，造成数据泄露
    unique_traj_ids = all_data['traj_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_traj_ids)
    
    split_idx = int(len(unique_traj_ids) * 0.8) # 80% 训练
    train_ids = unique_traj_ids[:split_idx]
    test_ids = unique_traj_ids[split_idx:]
    
    print(f"Splitting data: {len(train_ids)} trajectories for training, {len(test_ids)} trajectories for testing.")
    
    train_data = all_data[all_data['traj_id'].isin(train_ids)].copy()
    test_data = all_data[all_data['traj_id'].isin(test_ids)].copy()
    
    # 3. 训练集需要打乱 (shuffle)
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Train set size: {len(train_data)}, Test set size: {len(test_data)}")

    catboost_regressor = train(train_data, test_data)
    os.makedirs("models", exist_ok=True)  
    catboost_regressor.save_model("./models/stroke_model.cbm")
    
    best_threshold = evaluate(train_data, test_data, catboost_regressor)
    
    print("\n=" * 60)
    print("训练完成！模型已保存到 ./models/stroke_model.cbm")
    print(f"最佳阈值已保存到 best_threshold.txt: {best_threshold:.4f}")
    print("\n要进行预测，请运行: python predict.py")
    print("=" * 60)
    
    return best_threshold


if __name__ == "__main__":
    main()
