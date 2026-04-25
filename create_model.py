import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# MUST MATCH APP INPUT
X = np.random.rand(200, 19)
y = np.random.randint(0, 2, 200)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "fusion_model.pkl")

print("Model trained with 19 features")