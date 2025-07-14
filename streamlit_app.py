# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from traffic_model import (
    load_data,
    train_models,
    predict_future_6_batches,
    show_model_evaluation_dashboard
)
from holidays import holiday_dates

st.set_page_config(page_title="國道壅塞預測系統", layout="wide")
st.title("🚗 國道交通壅塞預測 (2025~2028)")

# === 上傳或指定檔案路徑 ===
with st.sidebar:
    st.header("📂 資料設定")
    volume_path = st.text_input("輸入車流量檔案路徑", value="vehicle_summary_10min_south.xlsx")
    speed_path = st.text_input("輸入速率資料檔案路徑", value="修正後_TravelSpeed_southV2.xlsx")

    if st.button("載入與訓練模型"):
        with st.spinner("載入與訓練中..."):
            df_model = load_data(volume_path, speed_path, holiday_dates)
            reg_model, clf_model, X_test, y_reg_test, y_clf_test = train_models()
            st.session_state.update({
                "trained": True,
                "reg_model": reg_model,
                "clf_model": clf_model,
                "X_test": X_test,
                "y_reg_test": y_reg_test,
                "y_clf_test": y_clf_test,
                "df_model": df_model
            })
        st.success("✅ 模型訓練完成！")

# === 評估頁籤 ===
if "trained" in st.session_state:
    st.subheader("📊 模型評估結果")
    show_model_evaluation_dashboard(
        st.session_state.y_reg_test,
        st.session_state.reg_model.predict(st.session_state.X_test),
        st.session_state.y_clf_test,
        st.session_state.clf_model.predict(st.session_state.X_test),
        st.session_state.clf_model.predict_proba(st.session_state.X_test)[:, 1]
    )

    # === 使用者輸入預測時間 ===
    st.subheader("🕒 輸入預測起始時間")
    col1, col2, col3, col4, col5 = st.columns(5)
    year = col1.selectbox("年份", list(range(2025, 2029)))
    month = col2.selectbox("月份", list(range(1, 13)))
    day = col3.selectbox("日期", list(range(1, 32)))
    hour = col4.selectbox("小時", list(range(0, 24)))
    minute = col5.selectbox("分鐘", [0, 10, 20, 30, 40, 50])

    input_time = datetime(year, month, day, hour, minute)
    date_str = input_time.strftime("%Y-%m-%d")

    if date_str in holiday_dates:
        st.error(f"⚠️ {date_str} 為國定假日，請選擇其他日期")
    elif input_time.weekday() >= 5:
        st.error(f"⚠️ {date_str} 是週末，請選擇平日")
    else:
        st.success(f"✅ 預測起始時間為：{input_time}")
        st.pyplot(predict_future_6_batches(input_time))
else:
    st.warning("⚠️ 請先載入並訓練模型")
