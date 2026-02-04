import catboost as ctb
import numpy as np

FEATURE_WINDOW_NUM = 7
_EPS = 1e-15


def build_features_from_points(points):
    """
    Build 24-dim feature vector from 5 points [[x,y],[x,y],[x,y],[x,y],[x,y]].
    points[0], points[1] = before current; points[2] = current; points[3], points[4] = after current.
    """
    if len(points) != FEATURE_WINDOW_NUM:
        raise ValueError(f"Expected {FEATURE_WINDOW_NUM} points, got {len(points)}")
    
    positions = [tuple(p) for p in points]
    
    # 当前点应为窗口中间的点，保证与 `stroke_model.to_features` 的 zero 对齐
    current = positions[FEATURE_WINDOW_NUM // 2]
    
    x_diff_features = []
    y_diff_features = []
    
    for i in range(1, FEATURE_WINDOW_NUM):
        lag_pos = positions[FEATURE_WINDOW_NUM - 1 - i]
        x_diff_features.append(lag_pos[0] - current[0])
        y_diff_features.append(lag_pos[1] - current[1])
        
    x_diff_inv_features = []
    y_diff_inv_features = []
    
    for i in range(1, FEATURE_WINDOW_NUM):
        lag_pos = positions[i]
        x_diff_inv_features.append(lag_pos[0] - current[0])
        y_diff_inv_features.append(lag_pos[1] - current[1])
   
    x_div_features = [
        x_diff_features[i] / (x_diff_inv_features[i] + _EPS)
        for i in range(FEATURE_WINDOW_NUM - 1)
    ]
    y_div_features = [
        y_diff_features[i] / (y_diff_inv_features[i] + _EPS)
        for i in range(FEATURE_WINDOW_NUM - 1)
    ]
    
    return np.array(
        [
            *x_diff_features,
            *x_diff_inv_features,
            *x_div_features,
            *y_diff_features,
            *y_diff_inv_features,
            *y_div_features,
        ],
        dtype=np.float32,
    )


class StrokePredictor:
    """
    Loads the stroke model and predicts stroke possibility from 5 points.
    Input format: [[x,y],[x,y],[x,y],[x,y],[x,y]] (before2, before1, current, after1, after2).
    """

    def __init__(self, path_model: str):
        self.model = ctb.CatBoostRegressor()
        self.model.load_model(path_model)
        self.model_path = path_model

    def predict(self, points) -> float:
        """
        Predict stroke possibility.
        :param points: list of 5 [x,y] pairs
        :return: possibility in [0, 1]
        """
        features = build_features_from_points(points)
        return float(self.model.predict(features.reshape(1, -1))[0])


if __name__ == "__main__":
    import os
    path_model = os.path.join(os.path.dirname(__file__), "models", "stroke_model.cbm")
    data = [
        [754.3333740234375, 164.66667175292997],
        [756.6666259765625, 168.3333282470703],
        [758.3333740234375, 172.66667175292997],
        [760.3333129882812, 174.3333282470703],
        [761.3333129882812, 168.3333282470703],
        [762.3333129882812, 163.3333282470703],
        [763.3333740234375, 159.3333282470703],
    ]
    predictor = StrokePredictor(path_model)
    features = build_features_from_points(data)
    if hasattr(predictor.model, "get_feature_count"):
        model_feature_count = predictor.model.get_feature_count()
    else:
        # get_feature_importance() returns array of length = feature count
        model_feature_count = len(predictor.model.get_feature_importance())
    build_feature_count = len(features)
    print(f"model feature count: {model_feature_count}")
    print(f"build_features_from_points feature count: {build_feature_count}")
    if model_feature_count == build_feature_count:
        result = predictor.predict(data)
        print(result)
    else:
        print("(skip predict: feature count mismatch)")
