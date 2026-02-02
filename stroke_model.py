import pandas as pd
import json
import numpy as np
import os
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt

# 数据文件路径
DATA_FILE = "Tennis-Stroke-Analysis-Data/output/training_segment.csv"

PREV_WINDOW_NUM = 2
AFTER_WINDOW_NUM = 2

def get_feature_cols(prev_window_num=PREV_WINDOW_NUM, after_window_num=AFTER_WINDOW_NUM):
    colnames_x = ['x_diff_{}'.format(i) for i in range(1, prev_window_num)] + \
                ['x_diff_inv_{}'.format(i) for i in range(1, after_window_num)] + \
                ["x_div_{}".format(i) for i in range(1, after_window_num)] #+ \
                #["x"]
    colnames_y = ['y_diff_{}'.format(i) for i in range(1, prev_window_num)] + \
                    ['y_diff_inv_{}'.format(i) for i in range(1, after_window_num)] + \
                    ["y_div_{}".format(i) for i in range(1, after_window_num)] #+ \
                    # ["y"]
    colnames = colnames_x + colnames_y #+ ["coord"]
    return colnames

def to_features(data, prev_window_num=PREV_WINDOW_NUM, after_window_num=AFTER_WINDOW_NUM):
    eps = 1e-15  # 防止除零错误
    data = data.copy()  # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    for i in range(1, prev_window_num):
        data.loc[:, 'x_lag_{}'.format(i)] = data['x'].shift(i)  # 创建一个新列，存储Y坐标的滞后值（即前几个时间点的Y坐标值）  data.loc[:, column_name]表示选择所有行和指定列,   .shift(i)：将这一列的数据向下移动i行
        data.loc[:, 'y_lag_{}'.format(i)] = data['y'].shift(i)
        data.loc[:, 'x_diff_{}'.format(i)] = data['x_lag_{}'.format(i)] - data['x']   # 计算当前点与滞后点的X坐标差值
        data.loc[:, 'y_diff_{}'.format(i)] = data['y_lag_{}'.format(i)] - data['y']


    for i in range(1, after_window_num):
        data.loc[:, 'x_lag_inv_{}'.format(i)] = data['x'].shift(-i)   # data['x'].shift(-i)：向上移动i行，获取未来的值    x_lag_inv_i, y_lag_inv_i: 存储未来i个时间步长的坐标值
        data.loc[:, 'y_lag_inv_{}'.format(i)] = data['y'].shift(-i) 
        data.loc[:, 'x_diff_inv_{}'.format(i)] = data['x_lag_inv_{}'.format(i)] - data['x']        # x_lag_inv_i, y_lag_inv_i: 存储未来i个时间步长的坐标值，利用未来信息（仅在特征工程中使用，实时预测时不可用）
        data.loc[:, 'y_diff_inv_{}'.format(i)] = data['y_lag_inv_{}'.format(i)] - data['y']


    for i in range(1, after_window_num):
        data.loc[:, 'x_div_{}'.format(i)] = data['x_diff_{}'.format(i)]/(data['x_diff_inv_{}'.format(i)] + eps)    # （过去坐标 - 当前坐标）/ （未来坐标 - 当前坐标）
        data.loc[:, 'y_div_{}'.format(i)] = data['y_diff_{}'.format(i)]/(data['y_diff_inv_{}'.format(i)] + eps)

    for i in range(1, prev_window_num):
        data = data[data['x_lag_{}'.format(i)].notna()]     #  保留x_lag_i列中非空（not null and not NaN）的行，移除由于shift操作产生的空值行（因为shift操作会在开始或结尾产生NaN值）
        
    for i in range(1, after_window_num):
        data = data[data['x_lag_inv_{}'.format(i)].notna()]
    
    data = data[data['x'].notna()] 
    return data

def __add_weight(pd_data, weight_map):   # 为数据添加权重，weight_map是一个字典，key是类别，value是权重
    pd_data["weight"] = pd_data["event_cls"].map(weight_map)
    return pd_data


def load_data(file_path, shuffle=True):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件 {file_path} 不存在。")
    
    # 1. 读取 CSV
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # 2. 过滤掉未核对的数据 (is_checked =！1 的是未标注数据，视为脏数据丢弃)
    if 'is_checked' in df.columns:
        original_len = len(df)
        df = df[df['is_checked'] == 1].copy()
        print(f"Filtered unchecked data: {original_len} -> {len(df)}")
    
    # 3. 解析标签 (event_cls)
    # hit_frames_global 格式可能是 "-1" 或 "49653" 或 "49653,51000"
    def check_is_hit(row):
        hit_str = str(row['hit_frames_global'])
        if hit_str == "-1" or hit_str == "":
            return 0
        hits = hit_str.split(',')
        # 如果当前帧号在击球帧列表中，则为正样本
        return 1 if str(row['frame_index']) in hits else 0
        
    df['event_cls'] = df.apply(check_is_hit, axis=1)
    
    # 4. 特征工程 (必须按 traj_id 分组处理，否则会在不同轨迹交界处产生错误的差分特征)
    # 先按 traj_id 和 frame_index 排序，确保时序正确
    df = df.sort_values(by=['traj_id', 'frame_index'])
    
    resdf = pd.DataFrame()
    
    # 使用 groupby 对每个轨迹单独计算特征
    # 注意：这里会比较耗时，但必须这样做以保证特征准确性
    grouped = df.groupby('traj_id')
    processed_list = []
    
    print("Processing features by trajectory group...")
    for traj_id, group in grouped:
        # 只有当轨迹长度足够计算窗口时才保留
        if len(group) > PREV_WINDOW_NUM + AFTER_WINDOW_NUM:
            processed_group = to_features(group)
            processed_list.append(processed_group)
            
    if len(processed_list) > 0:
        resdf = pd.concat(processed_list, ignore_index=True)
    else:
        raise ValueError("没有足够的数据生成特征，请检查 traj_id 分组或窗口大小配置。")

    # 6. 添加权重
    resdf = __add_weight(resdf, {1: 40, 0: 1})
    
    if shuffle:
        resdf = resdf.sample(frac=1, random_state=42).reset_index(drop=True)
        
    return resdf

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
    
    for threshold in np.arange(0.01, 0.99, 0.01):
        print(f'===> threshold: {threshold}')

        # 使用 sklearn 计算混淆矩阵
        pred_labels = (test_data["pred"] > threshold).astype(int)
        cm = confusion_matrix(test_data['event_cls'], pred_labels)
        tn, fp, fn, tp = cm.ravel()  # [[tn, fp], [fn, tp]]
        
        print(f'tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}, total: {tn + tp + fn + fp}')

        acc = (tn + tp) / (tn + tp + fn + fp) if (tn + tp + fn + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算F-beta分数 (beta^2 = 3, 所以召回率权重是精确率的3倍)
        beta_squared = 4
        f_beta = (1 + beta_squared) * precision * recall / (beta_squared * precision + recall) if (beta_squared * precision + recall) > 0 else 0
        
        print(f'accuracy: {acc}, recall: {recall}, precision: {precision}, f1: {f1}, f-beta: {f_beta}')
        
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
    
    # 绘制PR曲线并保存
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, label=f'PR Curve (AUC = {auc_pr:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('pr_curve.png')
    plt.close()
    print("PR curve saved as pr_curve.png")
    
    # 保存最佳阈值到文件，供 predict.py 使用
    with open('best_threshold.txt', 'w') as f:
        f.write(str(best_threshold))
    print(f"Best threshold saved to best_threshold.txt")
    
    return best_threshold


def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: Dataset {DATA_FILE} not found.")
        return None

    # 1. 加载所有数据 (先不打乱，以便这一步处理完特征后再做分割)
    # 注意：load_data 内部现在默认是 shuffle=True，但为了分割数据集，我们需要控制它
    # 修改 load_data 的 shuffle 参数，或者在 split 之前处理
    # 新版 load_data 若设为 False，返回的是按 traj_id 排序好的
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
    # 测试集保持时序 (不打乱)，以便后续可能的时序分析，或者 evaluate 里的逻辑
    # evaluate 函数其实并不严格依赖时序，因为它是逐点 predict，但保持有序是个好习惯
    
    print(f"Train set size: {len(train_data)}, Test set size: {len(test_data)}")

    catboost_regressor = train(train_data, test_data)
    catboost_regressor.save_model("stroke_model.cbm")
    
    best_threshold = evaluate(train_data, test_data, catboost_regressor)
    
    print("\n=" * 60)
    print("训练完成！模型已保存到 stroke_model.cbm")
    print(f"最佳阈值已保存到 best_threshold.txt: {best_threshold:.4f}")
    print("\n要进行预测，请运行: python predict.py")
    print("=" * 60)
    
    return best_threshold


if __name__ == "__main__":
    main()
