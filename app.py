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
# UI: Upload Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section">🎙 Voice Input</div>', unsafe_allow_html=True)
    audio_file = st.file_uploader(
        "Upload voice file",
        type=["wav", "ogg", "mp3", "m4a"]
    )

with col2:
    st.markdown('<div class="section">⌚ Wearable Data</div>', unsafe_allow_html=True)
    st.caption("Expected columns: heart_rate, eda, activity, sleep_hours")
    wearable_file = st.file_uploader("Upload CSV file", type=["csv"])

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict Mental Health Risk"):

    if audio_file is None:
        st.warning("Please upload a voice audio file.")

    elif wearable_file is None:
        st.warning("Please upload a wearable CSV file.")

    else:
        try:
            with st.spinner("Analyzing voice and wearable data..."):

                # ---- Save uploaded audio with original extension ----
                file_extension = audio_file.name.split(".")[-1]
                temp_audio_path = f"temp_audio.{file_extension}"

                with open(temp_audio_path, "wb") as f:
                    f.write(audio_file.read())

                # ---- Voice Feature Extraction ----
                voice_features = extract_voice_features(temp_audio_path)

                # ---- Wearable Feature Extraction ----
                wearable_df = pd.read_csv(wearable_file)
                wearable_features = wearable_df.drop(
                    'label', axis=1, errors='ignore'
                ).iloc[0].values

                # ---- Multimodal Fusion ----
                final_input = np.hstack((voice_features, wearable_features)).reshape(1, -1)

                # ---- Prediction ----
                prediction = model.predict(final_input)[0]
                confidence = model.predict_proba(final_input).max()

                # ---- Anxiety vs Depression (Rule-Based) ----
                heart_rate, eda, activity, sleep = wearable_features

                anxiety_score = int(heart_rate > 85) + int(eda > 2.5)
                depression_score = int(activity < 2000) + int(sleep < 6)

                anxiety_level = "HIGH" if anxiety_score >= 2 else "LOW"
                depression_level = "HIGH" if depression_score >= 2 else "LOW"

            # -----------------------------
            # Prediction Summary
            # -----------------------------
            st.markdown('<div class="section">📊 Prediction Summary</div>', unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("Overall Risk", "HIGH" if prediction == 1 else "LOW")

            with m2:
                st.metric("Confidence", f"{confidence:.2f}")

            with m3:
                st.metric("Anxiety Risk", anxiety_level)

            # -----------------------------
            # Risk Breakdown
            # -----------------------------
            st.markdown('<div class="section">🧠 Risk Breakdown</div>', unsafe_allow_html=True)
            st.write(f"**Anxiety Risk:** {anxiety_level}")
            st.write(f"**Depression Risk:** {depression_level}")

            # -----------------------------
            # Feature Importance
            # -----------------------------
            feature_names = [f"Voice_Feature_{i+1}" for i in range(15)] + \
                            ["Heart Rate", "EDA", "Activity", "Sleep"]

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            with st.expander("🔍 Feature Importance (Explainable AI)"):
                st.dataframe(importance_df.head(5))

            # -----------------------------
            # Trend Analysis
            # -----------------------------
            if len(wearable_df) > 1:
                with st.expander("📈 Mental Health Risk Trend Over Time"):
                    trend_scores = []
                    for i in range(len(wearable_df)):
                        wf = wearable_df.drop(
                            'label', axis=1, errors='ignore'
                        ).iloc[i].values
                        combined = np.hstack((voice_features, wf)).reshape(1, -1)
                        trend_scores.append(model.predict_proba(combined)[0][1])

                    st.line_chart(trend_scores)

            # -----------------------------
            # Disclaimer
            # -----------------------------
            st.markdown("---")
            st.caption(
                "⚠ Disclaimer: This application is for educational and research purposes only. "
                "It is not a medical diagnostic tool."
            )

        except Exception as e:
            st.error("An error occurred while processing your inputs.")
            st.exception(e)
