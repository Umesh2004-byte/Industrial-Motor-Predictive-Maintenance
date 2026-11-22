import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


# STEP 1: LOAD DATASETS

healthy = pd.read_csv("data/healthy_motor_data.csv")
faulty = pd.read_csv("data/faulty_motor_data.csv")

print("\nDatasets loaded successfully!")


# STEP 2: SELECT FEATURES

features = [
    "Voltage",
    "Current",
    "Temperature",
    "Vibration",
    "Speed",
    "Power"
]

X_healthy = healthy[features]


# STEP 3: TRAIN ISOLATION FOREST MODEL

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

model.fit(X_healthy)

print("\nIsolation Forest model trained successfully!")


# STEP 4: COMBINE HEALTHY AND FAULTY DATA

healthy["Actual_Label"] = 0
faulty["Actual_Label"] = 1

combined_data = pd.concat(
    [healthy, faulty],
    ignore_index=True
)

X_test = combined_data[features]


# STEP 5: PREDICT ANOMALIES

predictions = model.predict(X_test)

# Isolation Forest:
# 1  = Normal
# -1 = Anomaly

combined_data["Prediction"] = predictions

combined_data["Predicted_Label"] = np.where(
    predictions == -1, 1, 0
)

combined_data["Detection_Status"] = np.where(
    predictions == -1,
    "Anomaly Detected",
    "Normal"
)

print("\nAnomaly detection completed!")


# STEP 6: DISPLAY RESULTS

print("\nPREDICTION SUMMARY\n")

print(
    combined_data.groupby(
        ["Condition", "Detection_Status"],
        dropna=False
    ).size()
)

print("\nCONFUSION MATRIX\n")

print(
    confusion_matrix(
        combined_data["Actual_Label"],
        combined_data["Predicted_Label"]
    )
)

print("\nCLASSIFICATION REPORT\n")

print(
    classification_report(
        combined_data["Actual_Label"],
        combined_data["Predicted_Label"],
        target_names=["Normal", "Anomaly"]
    )
)


# STEP 7: SAVE PREDICTION RESULTS

combined_data.to_csv(
    "data/anomaly_detection_results.csv",
    index=False
)

print("\nPrediction results saved successfully!")


# STEP 8: SAVE TRAINED MODEL

joblib.dump(
    model,
    "models/isolation_forest_model.pkl"
)

print("\nTrained model saved successfully!")


# STEP 9: VISUALIZE ANOMALIES

plt.figure(figsize=(12, 6))

normal_data = combined_data[
    combined_data["Predicted_Label"] == 0
]

anomaly_data = combined_data[
    combined_data["Predicted_Label"] == 1
]

plt.scatter(
    normal_data["Time"],
    normal_data["Current"],
    label="Normal",
    alpha=0.6
)

plt.scatter(
    anomaly_data["Time"],
    anomaly_data["Current"],
    label="Anomaly",
    alpha=0.6
)

plt.title("Isolation Forest - Motor Anomaly Detection")
plt.xlabel("Time (seconds)")
plt.ylabel("Current (A)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/anomaly_detection.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nModule 4 completed successfully!")