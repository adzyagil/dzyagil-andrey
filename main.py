"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import os

np.random.seed(322)


def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    predictions.to_csv(submission_path, index=False)
    print(f"Submission файл сохранен: {submission_path}")
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # === ЗАГРУЗКА ===
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    train["dt"] = pd.to_datetime(train["dt"])
    test["dt"] = pd.to_datetime(test["dt"])
    train = train.sort_values(["product_id", "dt"])

    # === БЕЙЗЛАЙН + ГАРАНТИРОВАННОЕ УЛУЧШЕНИЕ ===
    train["center"] = (train.price_p05 + train.price_p95) / 2
    train["lower_delta"] = train.center - train.price_p05
    train["upper_delta"] = train.price_p95 - train.center
    train["log_center"] = np.log1p(train.center)
    train["log_ld"] = np.log1p(train.lower_delta)
    train["log_ud"] = np.log1p(train.upper_delta)

    for df in [train, test]:
        df["dow"] = df.dt.dt.weekday
        df["week"] = df.dt.dt.isocalendar().week.astype(int)
        df["month"] = df.dt.dt.month
        df["is_weekend"] = (df["dow"] >= 5).astype(int)

    windows = [3, 7, 14, 28]
    for w in windows:
        train[f"c_mean_{w}"] = train.groupby("product_id")["center"].shift(1).rolling(w).mean()
        train[f"c_std_{w}"] = train.groupby("product_id")["center"].shift(1).rolling(w).std()
        train[f"ld_mean_{w}"] = train.groupby("product_id")["lower_delta"].shift(1).rolling(w).mean()
        train[f"ud_mean_{w}"] = train.groupby("product_id")["upper_delta"].shift(1).rolling(w).mean()

    roll_cols = [c for c in train.columns if "_mean_" in c or "_std_" in c]
    train[roll_cols] = train[roll_cols].fillna(train[roll_cols].median())

    cat_cols = [
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id"
    ]

    for c in cat_cols:
        le = LabelEncoder()
        train[c] = le.fit_transform(train[c].astype(str))
        test[c] = le.transform(test[c].astype(str))

    features = [
        "dow","week","month","is_weekend",
        "n_stores","holiday_flag","activity_flag",
        "precpt","avg_temperature","avg_humidity","avg_wind_level"
    ] + roll_cols + cat_cols

    split_date = train.dt.quantile(0.85)
    tr = train[train.dt <= split_date]
    va = train[train.dt > split_date]

    def fit_quantile(target, q):
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=3000,
            learning_rate=0.02,
            num_leaves=96,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=322
        )
        model.fit(
            tr[features], tr[target],
            eval_set=[(va[features], va[target])],
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )
        return model

    # === КЛЮЧЕВОЕ УЛУЧШЕНИЕ: асимметричные квантили (0.95 / 0.85) ===
    center_models = {0.5: fit_quantile("log_center", 0.5)}
    ld_models = {0.95: fit_quantile("log_ld", 0.95)}   # ← 0.95 для нижней границы
    ud_models = {0.85: fit_quantile("log_ud", 0.85)}   # ← 0.85 для верхней границы

    def predict_q(models, df):
        return {q: np.expm1(m.predict(df[features])) for q,m in models.items()}

    # Прогноз на test
    last_roll = train.sort_values("dt").groupby("product_id")[roll_cols].last().reset_index()
    test = test.merge(last_roll, on="product_id", how="left")
    test[roll_cols] = test[roll_cols].fillna(train[roll_cols].median())

    test_center_q = predict_q(center_models, test)
    test_ld_q = predict_q(ld_models, test)
    test_ud_q = predict_q(ud_models, test)

    test["price_p05"] = test_center_q[0.5] - test_ld_q[0.95]   # ← 0.95
    test["price_p95"] = test_center_q[0.5] + test_ud_q[0.85]   # ← 0.85

    eps = 1e-3
    test["price_p05"] = np.maximum(test["price_p05"], 0)
    test["price_p95"] = np.maximum(test["price_p95"], test["price_p05"] + eps)

    submission = test[["row_id", "price_p05", "price_p95"]].copy()

    # Создание submission
    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()