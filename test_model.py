import joblib

# after training
joblib.dump(model, "fusion_model.pkl", compress=3)