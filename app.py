import streamlit as st
import numpy as np
import pandas as pd
import joblib
import librosa
import os

# -----------------------------
# Page Configuration & Styling
# -----------------------------
st.set_page_config(page_title="AI Mental Health Monitor", layout="centered")

st.markdown("""
<style>
.title {font-size:28px; font-weight:700; margin-bottom:10px;}
.section {font-size:20px; font-weight:600; margin-top:20px;}
.box {padding:15px; border-radius:10px; background-color:#f0f2f6; margin-top:10px;}
.good {color:#2ecc71; font-weight:600;}
.mild {color:#f1c40f; font-weight:600;}
.high {color:#e74c3c; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown('<div class="title">🧠 AI Mental Health Monitoring System</div>', unsafe_allow_html=True)
st.write("Multimodal analysis using voice and wearable data")

# -----------------------------
# Load trained fusion model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fusion_model.pkl")
model = joblib.load(MODEL_PATH)

# -----------------------------
# Helper: Extract voice features
# -----------------------------
def extract_voice_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)

    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch_mean = np.nanmean(pitch)

    energy = librosa.feature.rms(y=y)
    energy_mean = np.mean(energy)

    return np.hstack((mfcc_mean, pitch_mean, energy_mean))

# -----------------------------
# UI: Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section">🎙 Voice Input</div>', unsafe_allow_html=True)
    audio_file = st.file_uploader(
        "Upload voice file",
        type=["wav", "ogg", "mp3", "m4a"]
    )

with col2:
    st.markdown('<div class="section">⌚ Wearable Data (Quick Select)</div>', unsafe_allow_html=True)

    wearable_option = st.selectbox(
        "Choose wearable data scenario",
        [
            "🟢 Normal / Healthy",
            "🟡 Mild Stress",
            "🔴 High Anxiety",
            "🔵 Depression-like Pattern",
            "⚙ Custom (Manual Input)"
        ]
    )

# -----------------------------
# Wearable Data Logic (UI ONLY)
# -----------------------------
if wearable_option == "🟢 Normal / Healthy":
    heart_rate, eda, activity, sleep = 72, 1.2, 3500, 7.5
    st.markdown('<p class="good">Healthy physiological pattern</p>', unsafe_allow_html=True)

elif wearable_option == "🟡 Mild Stress":
    heart_rate, eda, activity, sleep = 82, 2.0, 2600, 6.8
    st.markdown('<p class="mild">Mild stress indicators</p>', unsafe_allow_html=True)

elif wearable_option == "🔴 High Anxiety":
    heart_rate, eda, activity, sleep = 95, 3.5, 2200, 5.8
    st.markdown('<p class="high">High anxiety indicators</p>', unsafe_allow_html=True)

elif wearable_option == "🔵 Depression-like Pattern":
    heart_rate, eda, activity, sleep = 68, 1.8, 1200, 5.5
    st.markdown('<p class="mild">Low activity & sleep pattern</p>', unsafe_allow_html=True)

elif wearable_option == "⚙ Custom (Manual Input)":
    heart_rate = st.slider("Heart Rate (bpm)", 50, 120, 80)
    eda = st.slider("EDA / Stress Level", 0.5, 5.0, 2.0)
    activity = st.slider("Daily Activity (steps)", 500, 6000, 2500)
    sleep = st.slider("Sleep Duration (hours)", 3.0, 9.0, 6.5)

wearable_features = np.array([heart_rate, eda, activity, sleep])
wearable_df = pd.DataFrame(
    [wearable_features],
    columns=["heart_rate", "eda", "activity", "sleep_hours"]
)

st.info(
    f"Using wearable values → HR: {heart_rate}, "
    f"EDA: {eda}, Activity: {activity}, Sleep: {sleep} hrs"
)

# -----------------------------
# Advanced Options (CSV Hidden)
# -----------------------------
with st.expander("⚙ Advanced Options (Upload Wearable CSV)"):
    wearable_file = st.file_uploader("Upload wearable CSV", type=["csv"])

    if wearable_file is not None:
        wearable_df = pd.read_csv(wearable_file)

        required_cols = {"heart_rate", "eda", "activity", "sleep_hours"}
        if not required_cols.issubset(wearable_df.columns):
            st.error("CSV must contain: heart_rate, eda, activity, sleep_hours")
            st.stop()

        wearable_features = wearable_df.iloc[0].values
        st.success("CSV data loaded successfully")

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict Mental Health Risk"):

    if audio_file is None:
        st.warning("Please upload a voice audio file.")

    else:
        try:
            with st.spinner("Analyzing voice and wearable data..."):

                file_extension = audio_file.name.split(".")[-1]
                temp_audio_path = f"temp_audio.{file_extension}"

                with open(temp_audio_path, "wb") as f:
                    f.write(audio_file.read())

                voice_features = extract_voice_features(temp_audio_path)

                final_input = np.hstack((voice_features, wearable_features)).reshape(1, -1)

                prediction = model.predict(final_input)[0]
                confidence = model.predict_proba(final_input).max()

                anxiety_score = int(heart_rate > 85) + int(eda > 2.5)
                depression_score = int(activity < 2000) + int(sleep < 6)

                anxiety_level = "HIGH" if anxiety_score >= 2 else "LOW"
                depression_level = "HIGH" if depression_score >= 2 else "LOW"

            st.markdown('<div class="section">📊 Prediction Summary</div>', unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("Overall Risk", "HIGH" if prediction == 1 else "LOW")
            with m2:
                st.metric("Confidence", f"{confidence:.2f}")
            with m3:
                st.metric("Anxiety Risk", anxiety_level)

            st.markdown('<div class="section">🧠 Risk Breakdown</div>', unsafe_allow_html=True)
            st.write(f"**Anxiety Risk:** {anxiety_level}")
            st.write(f"**Depression Risk:** {depression_level}")

            feature_names = [f"Voice_Feature_{i+1}" for i in range(15)] + \
                            ["Heart Rate", "EDA", "Activity", "Sleep"]

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            with st.expander("🔍 Feature Importance (Explainable AI)"):
                st.dataframe(importance_df.head(5))

            if len(wearable_df) > 1:
                with st.expander("📈 Mental Health Risk Trend Over Time"):
                    trend_scores = []
                    for i in range(len(wearable_df)):
                        wf = wearable_df.iloc[i].values
                        combined = np.hstack((voice_features, wf)).reshape(1, -1)
                        trend_scores.append(model.predict_proba(combined)[0][1])
                    st.line_chart(trend_scores)

            st.markdown("---")
            st.caption(
                "⚠ Disclaimer: This application is for educational and research purposes only. "
                "It is not a medical diagnostic tool."
            )

        except Exception as e:
            st.error("An error occurred while processing your inputs.")
            st.exception(e)
