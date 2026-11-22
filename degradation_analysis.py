import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


# STEP 1: LOAD DATASETS

healthy = pd.read_csv("data/healthy_motor_data.csv")
faulty = pd.read_csv("data/faulty_motor_data.csv")

print("\nDatasets loaded successfully!")


# STEP 2: SELECT BEARING DETERIORATION DATA

bearing = faulty[
    faulty["Condition"] == "Bearing Deterioration"
].copy()

print("\nBearing deterioration data loaded!")


# STEP 3: CALCULATE HEALTHY BASELINE

parameters = [
    "Vibration",
    "Temperature",
    "Current"
]

healthy_mean = healthy[parameters].mean()
healthy_std = healthy[parameters].std()

print("\nHealthy Motor Baseline:")
print(healthy_mean.round(3))


# STEP 4: CALCULATE STANDARDIZED DEVIATION

for parameter in parameters:

    bearing[parameter + "_Zscore"] = (
        bearing[parameter] - healthy_mean[parameter]
    ) / healthy_std[parameter]


# STEP 5: CALCULATE DEGRADATION INDEX

zscore_columns = [
    "Vibration_Zscore",
    "Temperature_Zscore",
    "Current_Zscore"
]

bearing["Degradation_Index"] = (
    bearing[zscore_columns]
    .clip(lower=0)
    .mean(axis=1)
)

print("\nDegradation Index calculated successfully!")


# STEP 6: CALCULATE ROLLING AVERAGE

bearing["Rolling_Degradation"] = (
    bearing["Degradation_Index"]
    .rolling(window=10, min_periods=1)
    .mean()
)


# STEP 7: TRAIN LINEAR REGRESSION FOR TREND

X = bearing[["Time"]]
y = bearing["Rolling_Degradation"]

trend_model = LinearRegression()

trend_model.fit(X, y)

bearing["Predicted_Degradation"] = trend_model.predict(X)


# STEP 8: ESTIMATE WARNING THRESHOLD CROSSING

WARNING_THRESHOLD = 3.0

slope = trend_model.coef_[0]
intercept = trend_model.intercept_

print("\nDEGRADATION ANALYSIS RESULTS")

print("Trend Slope:", round(slope, 4))
print("Current Degradation Index:",
      round(bearing["Rolling_Degradation"].iloc[-1], 3))

if slope > 0:

    crossing_time = (
        WARNING_THRESHOLD - intercept
    ) / slope

    if crossing_time > bearing["Time"].max():

        print(
            "Estimated warning threshold crossing:",
            round(crossing_time, 2),
            "seconds"
        )

    else:

        print(
            "Warning threshold has already been reached "
            "or the fitted trend does not provide a future crossing."
        )

else:

    print(
        "No increasing degradation trend detected. "
        "Future threshold crossing cannot be estimated."
    )


# STEP 9: SAVE ANALYSIS RESULTS

bearing.to_csv(
    "data/bearing_degradation_analysis.csv",
    index=False
)

print("\nDegradation analysis results saved successfully!")


# STEP 10: VISUALIZE DEGRADATION TREND

plt.figure(figsize=(12, 6))

plt.plot(
    bearing["Time"],
    bearing["Degradation_Index"],
    label="Degradation Index",
    alpha=0.5
)

plt.plot(
    bearing["Time"],
    bearing["Rolling_Degradation"],
    label="Rolling Average",
    linewidth=2
)

plt.plot(
    bearing["Time"],
    bearing["Predicted_Degradation"],
    label="Linear Trend",
    linestyle="--",
    linewidth=2
)

plt.axhline(
    y=WARNING_THRESHOLD,
    linestyle=":",
    label="Illustrative Warning Threshold"
)

plt.title("Motor Bearing - Degradation Analysis")
plt.xlabel("Time (seconds)")
plt.ylabel("Degradation Index")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "outputs/bearing_degradation_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nMODULE 5 COMPLETED SUCCESSFULLY!")