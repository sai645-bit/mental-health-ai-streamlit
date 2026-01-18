import streamlit as st
import numpy as np
import pandas as pd
import joblib
import librosa
import os

# -----------------------------
# Custom UI Styling
# -----------------------------
st.set_page_config(page_title="AI Mental Health Monitor", layout="centered")

st.markdown("""
<style>
.title {font-size:28px; font-weight:700; margin-bottom:10px;}
.section {font-size:20px; font-weight:600; margin-top:20px;}
.box {padding:15px; border-radius:10px; background-color:#f0f2f6; margin-top:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 AI Mental Health Monitoring System</div>', unsafe_allow_html=True)
st.write("Multimodal analysis using voice and wearable data")


# -----------------------------
# Title
# -----------------------------
st.markdown('<div class="title">🧠 AI Mental Health Monitoring System</div>', unsafe_allow_html=True)
st.write("Multimodal analysis using voice and wearable data")

# -----------------------------
# Load trained fusion model (SAFE PATH)
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

    features = np.hstack((mfcc_mean, pitch_mean, energy_mean))
    return features  # 15 features

# -----------------------------
# UI: Upload Inputs
# -----------------------------
st.markdown('<div class="section">🎙 Step 1: Upload Voice Sample</div>', unsafe_allow_html=True)
audio_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

st.markdown('<div class="section">⌚ Step 2: Upload Wearable Data</div>', unsafe_allow_html=True)
st.markdown("Expected columns: **heart_rate, eda, activity, sleep_hours**")
wearable_file = st.file_uploader("Upload wearable CSV file", type=["csv"])

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict Mental Health Risk"):

    if audio_file is None:
        st.warning("Please upload a voice audio file (.wav).")

    elif wearable_file is None:
        st.warning("Please upload a wearable CSV file.")

    else:
        try:
            # ---- Voice Feature Extraction ----
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file.read())

            voice_features = extract_voice_features("temp_audio.wav")

            # ---- Wearable Data ----
            wearable_df = pd.read_csv(wearable_file)

            if 'label' in wearable_df.columns:
                wearable_features = wearable_df.drop('label', axis=1).iloc[0].values
            else:
                wearable_features = wearable_df.iloc[0].values

            # ---- Multimodal Fusion ----
            final_input = np.hstack((voice_features, wearable_features)).reshape(1, -1)

            # ---- Prediction ----
            prediction = model.predict(final_input)[0]
            confidence = model.predict_proba(final_input).max()

            # -----------------------------
            # Result Display
            # -----------------------------
            st.markdown('<div class="box">', unsafe_allow_html=True)

            if prediction == 1:
                st.error("⚠ Mental Health Risk: HIGH")
            else:
                st.success("✅ Mental Health Risk: LOW")

            st.info(f"Confidence: {confidence:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

            # -----------------------------
            # Anxiety vs Depression (Rule-Based)
            # -----------------------------
            heart_rate = wearable_features[0]
            eda = wearable_features[1]
            activity = wearable_features[2]
            sleep = wearable_features[3]

            anxiety_score = 0
            depression_score = 0

            if heart_rate > 85:
                anxiety_score += 1
            if eda > 2.5:
                anxiety_score += 1

            if activity < 2000:
                depression_score += 1
            if sleep < 6:
                depression_score += 1

            anxiety_level = "HIGH" if anxiety_score >= 2 else "LOW"
            depression_level = "HIGH" if depression_score >= 2 else "LOW"

            st.markdown('<div class="section">🧠 Risk Breakdown</div>', unsafe_allow_html=True)
            st.write(f"**Anxiety Risk:** {anxiety_level}")
            st.write(f"**Depression Risk:** {depression_level}")

            # -----------------------------
            # Feature Importance (Explainable AI)
            # -----------------------------
            voice_feature_names = [f"Voice_Feature_{i+1}" for i in range(15)]
            wearable_feature_names = ["Heart Rate", "EDA", "Activity", "Sleep"]
            all_feature_names = voice_feature_names + wearable_feature_names

            importance_df = pd.DataFrame({
                "Feature": all_feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.markdown('<div class="section">🔍 Feature Importance</div>', unsafe_allow_html=True)
            st.dataframe(importance_df.head(5))

            # -----------------------------
            # Trend Analysis
            # -----------------------------
            if len(wearable_df) > 1:
                st.markdown('<div class="section">📈 Risk Trend Over Time</div>', unsafe_allow_html=True)

                trend_scores = []
                for i in range(len(wearable_df)):
                    wf = wearable_df.drop('label', axis=1, errors='ignore').iloc[i].values
                    combined = np.hstack((voice_features, wf)).reshape(1, -1)
                    trend_scores.append(model.predict_proba(combined)[0][1])

                st.line_chart(trend_scores)

        except Exception as e:
            st.error("An error occurred while processing your inputs.")
            st.exception(e)
