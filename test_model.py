import joblib

model = joblib.load("fusion_model.pkl")
print(model.n_features_in_)