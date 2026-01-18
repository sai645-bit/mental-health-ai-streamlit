import streamlit as st
import numpy as np
import pandas as pd
import joblib
import librosa

# -----------------------------
# Load trained fusion model
# -----------------------------
model = joblib.load("fusion_model.pkl")

st.set_page_config(page_title="AI Mental Health Monitor", layout="centered")

st.title("🧠 AI-Based Mental Health Risk Detection")
st.subheader("Multimodal Analysis using Voice + Wearable Data")

st.markdown("""
This system analyzes **speech characteristics** and **wearable sensor data**  
to estimate early risk of **depression or anxiety**.

⚠ *For educational purposes only. Not a medical diagnosis.*
""")

# -----------------------------
# Helper: Extract voice features
# -----------------------------
def extract_voice_features(audio_path, n_mfcc=13):
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # 1) MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)  # shape: (n_mfcc,)

    # 2) Pitch (fundamental frequency)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch_mean = np.nanmean(pitch)

    # 3) Energy (RMS)
    energy = librosa.feature.rms(y=y)
    energy_mean = np.mean(energy)

    # Combine: if you trained with 15 voice features, this gives 13 + 1 + 1 = 15
    features = np.hstack((mfcc_mean, pitch_mean, energy_mean))
    return features


# -----------------------------
# UI: Uploads
# -----------------------------
st.header("🎙 Upload Voice Sample (.wav)")
audio_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

st.header("⌚ Upload Wearable Data (CSV)")
st.markdown("Expected columns: **heart_rate, eda, activity, sleep_hours**")
wearable_file = st.file_uploader("Upload wearable CSV file", type=["csv"])


# -----------------------------
# Predict
# -----------------------------
if st.button("🔍 Predict Mental Health Risk"):

    if audio_file is None:
        st.warning("Please upload a voice audio file (.wav).")
    elif wearable_file is None:
        st.warning("Please upload a wearable CSV file.")
    else:
        try:
            # ---- 1) Extract voice features from uploaded audio ----
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file.read())

            voice_features = extract_voice_features("temp_audio.wav")

            # ---- 2) Read wearable CSV ----
            wearable_df = pd.read_csv(wearable_file)

            # If 'label' column exists, drop it; otherwise use all columns
            if 'label' in wearable_df.columns:
                wearable_features = wearable_df.drop('label', axis=1).iloc[0].values
            else:
                wearable_features = wearable_df.iloc[0].values

            # ---- 3) Fuse features (voice + wearable) ----
            final_input = np.hstack((voice_features, wearable_features)).reshape(1, -1)

            # ---- 4) Predict ----
            prediction = model.predict(final_input)[0]
            confidence = model.predict_proba(final_input).max()

            st.markdown("---")
            if prediction == 1:
                st.error("⚠ Mental Health Risk: **HIGH**")
            else:
                st.success("✅ Mental Health Risk: **LOW**")

            st.info(f"Confidence: {confidence:.2f}")

        except Exception as e:
            st.error("An error occurred while processing your inputs.")
            st.exception(e)
