# 國道交通預測 - 完整模組化程式

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ======== 全域變數 ======== #
reg_model = None
clf_model = None
df_model = None
holiday_dates = set()

# ======== 函式：取得紅綠燈建議 ======== #
def get_signal_control(speed):
    if speed < 20:
        return "紅燈建議：300 秒（5 分鐘）"
    elif speed >= 60:
        return "紅燈建議：0 秒（不需控制）"
    else:
        seconds = int((60 - speed) / 40 * 300)
        return f"紅燈建議：{seconds} 秒"

# ======== 函式：讀取與處理資料 ======== #
def load_data(volume_path, speed_path, holidays):
    global df_model, holiday_dates
    holiday_dates = set(holidays)

    volume_df = pd.read_csv(volume_path)
    speed_df = pd.read_csv(speed_path)

    volume_df["時間"] = pd.to_datetime(volume_df["十分鐘"]).dt.floor("10min")
    speed_df["時間"] = pd.to_datetime(speed_df["DataCollectTime"]).dt.floor("10min").dt.tz_localize(None)

    volume_df["日期"] = volume_df["時間"].dt.date
    speed_df["日期"] = speed_df["時間"].dt.date

    merged_df = pd.merge(volume_df, speed_df[["時間", "TravelSpeed", "CongestionLevel"]], on="時間", how="left")
    merged_df = merged_df[~merged_df["日期"].astype(str).isin(holiday_dates)]

    merged_df["hour"] = merged_df["時間"].dt.hour
    merged_df["minute"] = merged_df["時間"].dt.minute
    merged_df["星期"] = merged_df["時間"].dt.dayofweek

    filtered_df = merged_df[merged_df["星期"].isin([0, 1, 2, 3, 4])].copy()
    df = filtered_df.reset_index(drop=True)

    df["Speed_MA5"] = df["TravelSpeed"].rolling(5).mean()
    df["Speed_Slope"] = df["TravelSpeed"].diff()
    df["Weekday"] = df["時間"].dt.weekday
    df["Minute"] = df["時間"].dt.hour * 60 + df["時間"].dt.minute
    df["FutureSpeed"] = df["TravelSpeed"].shift(-1)
    df["FutureCongested"] = (df["FutureSpeed"] < 60).astype(int)

    df_model = df.dropna(subset=["TravelSpeed", "Speed_MA5", "Speed_Slope", "FutureSpeed"])
    return df_model

# ======== 函式：訓練模型 ======== #
def train_models():
    global reg_model, clf_model, df_model
    features = ["TravelSpeed", "Speed_MA5", "Speed_Slope", "Minute", "Weekday"]
    X = df_model[features]
    y_reg = df_model["FutureSpeed"]
    y_clf = df_model["FutureCongested"]

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [8, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    reg_gs = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=cv,
                          scoring='neg_mean_squared_error', n_jobs=-1)
    clf_gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=cv,
                          scoring='accuracy', n_jobs=-1)

    reg_gs.fit(X_train, y_reg_train)
    clf_gs.fit(X_train, y_clf_train)

    reg_model = reg_gs.best_estimator_
    clf_model = clf_gs.best_estimator_

    return reg_model, clf_model, X_test, y_reg_test, y_clf_test

# ======== 函式：模型預測視覺化 ======== #
def show_model_evaluation_dashboard(y_reg_test, y_reg_pred, y_clf_test, y_clf_pred, y_clf_prob):
    # 關閉舊圖，避免重複產生多視窗
    plt.close('all')
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(6, 4)

    mse = mean_squared_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    r2 = r2_score(y_reg_test, y_reg_pred)
    accuracy = accuracy_score(y_clf_test, y_clf_pred)
    precision = precision_score(y_clf_test, y_clf_pred, zero_division=0)
    recall = recall_score(y_clf_test, y_clf_pred, zero_division=0)
    f1 = f1_score(y_clf_test, y_clf_pred, zero_division=0)
    auc = roc_auc_score(y_clf_test, y_clf_prob)
    cm = confusion_matrix(y_clf_test, y_clf_pred)

    ax1 = fig.add_subplot(gs[0:3, 0:2])
    ax1.scatter(y_reg_test, y_reg_pred, alpha=0.5, edgecolor='k', s=20)
    ax1.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--')
    ax1.set_xlabel("實際車速 (km/h)")
    ax1.set_ylabel("預測車速 (km/h)")
    ax1.set_title("回歸預測結果")
    ax1.grid(True)
    ax1.text(0.95, 0.05, f"RMSE={rmse:.2f}\nR2={r2:.2f}", transform=ax1.transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8), ha='right', va='bottom')

    ax2 = fig.add_subplot(gs[0:3, 2:4])
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False,
                xticklabels=['正常', '壅塞'], yticklabels=['正常', '壅塞'], ax=ax2)
    ax2.set_xlabel("預測結果")
    ax2.set_ylabel("實際情況")
    ax2.set_title(f"壅塞分類結果\nAccuracy={accuracy:.2%}, AUC={auc:.2f}")

    ax3 = fig.add_subplot(gs[3:6, 0:2])
    ax3.axis('off')
    ax4 = fig.add_subplot(gs[3:6, 2:4])
    ax4.axis('off')

    text_left = (
        f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR2: {r2:.2f}"
    )
    text_right = (
        f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1: {f1:.2f}\nAUC: {auc:.2f}"
    )

    ax3.text(0, 1, text_left, fontsize=10, va='top', family='Microsoft JhengHei')
    ax4.text(0, 1, text_right, fontsize=10, va='top', family='Microsoft JhengHei')

    fig.suptitle("模型預測與分類效能總覽", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

# ======== 函式：互動預測與繪圖 ======== #
def predict_future_6_batches(input_time):
    plt.close('all')  # 每次預測前清除舊圖，避免重複視窗
    future_times = []
    predicted_speeds = []
    predicted_congestions = []
    signal_controls = []
    current_time = input_time

    for i in range(6):
        weekday = current_time.weekday()
        minute_of_day = current_time.hour * 60 + current_time.minute
        matched = df_model[(df_model["Weekday"] == weekday) & (df_model["Minute"] == minute_of_day)]

        if matched.empty:
            future_times.append(current_time.strftime('%H:%M'))
            predicted_speeds.append(np.nan)
            predicted_congestions.append(None)
            signal_controls.append("N/A")
        else:
            avg_speed = matched["TravelSpeed"].mean()
            avg_ma5 = matched["Speed_MA5"].mean()
            avg_slope = matched["Speed_Slope"].mean()
            input_features = np.array([[avg_speed, avg_ma5, avg_slope, minute_of_day, weekday]])
            predicted_speed = reg_model.predict(input_features)[0]
            predicted_congested = clf_model.predict(input_features)[0]
            signal_control = get_signal_control(predicted_speed)

            future_times.append(current_time.strftime('%H:%M'))
            predicted_speeds.append(predicted_speed)
            predicted_congestions.append(predicted_congested)
            signal_controls.append(signal_control)

        current_time += timedelta(minutes=10)

    fig = plt.figure(figsize=(13, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    ax0 = plt.subplot(gs[0])
    ax0.axis('off')
    columns = ["時間", "車速 (km/h)", "壅塞狀態", "紅燈建議"]
    table_data = []
    highlight_rows = []

    for idx, (t, s, c, ctrl) in enumerate(zip(future_times, predicted_speeds, predicted_congestions, signal_controls)):
        status = "壅塞" if c == 1 else "正常" if c == 0 else "N/A"
        speed_str = f"{s:.2f}" if not np.isnan(s) else "N/A"
        table_data.append([t, speed_str, status, ctrl])
        if c == 1:
            highlight_rows.append(idx + 1)

    table = ax0.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.scale(1.1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    col_widths = [0.15, 0.15, 0.15, 0.55]
    for col_idx, width in enumerate(col_widths):
        for row_idx in range(len(table_data) + 1):
            cell = table[row_idx, col_idx]
            cell.set_width(width)
    for row in highlight_rows:
        for col in range(len(columns)):
            cell = table[row, col]
            cell.set_facecolor("#ffcccc")
            cell.get_text().set_color("red")
            cell.get_text().set_weight("bold")

    ax1 = plt.subplot(gs[1])
    ax1.plot(future_times, predicted_speeds, marker='o', label='預測車速', color='blue')
    for i, (t, c) in enumerate(zip(future_times, predicted_congestions)):
        if c == 1:
            ax1.scatter(t, predicted_speeds[i], color='red', s=80, label='壅塞點' if i == 0 else "")
            ax1.annotate(f"{t}", (t, predicted_speeds[i]), textcoords="offset points", xytext=(0,10), ha='center', color='red', fontsize=10)

    ax1.set_title(f"{input_time.date()} {input_time.strftime('%H:%M')} 起未來一小時預測")
    ax1.set_xlabel("時間")
    ax1.set_ylabel("車速 (km/h)")
    ax1.grid(True)
    ax1.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
