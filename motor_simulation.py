import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. MOTOR SPECIFICATIONS
# ==========================================

MOTOR_POWER = 3.7          # kW (5 HP)
RATED_VOLTAGE = 415        # Volts
RATED_CURRENT = 6          # Amperes (illustrative)
RATED_SPEED = 1440         # RPM
FREQUENCY = 50              # Hz

# ==========================================
# 2. SIMULATION SETTINGS
# ==========================================

np.random.seed(42)

duration = 100
time = np.arange(0, duration, 1)

# ==========================================
# 3. HEALTHY MOTOR PARAMETERS
# ==========================================

voltage = np.random.normal(415, 2, duration)

current = np.random.normal(6, 0.15, duration)

temperature = (
    45 + 0.03 * time +
    np.random.normal(0, 0.3, duration)
)

vibration = np.random.normal(2, 0.1, duration)

speed = np.random.normal(1440, 2, duration)

power = (
    np.sqrt(3) * voltage * current * 0.85 / 1000
)

# ==========================================
# 4. CREATE DATAFRAME
# ==========================================

data = pd.DataFrame({
    "Time": time,
    "Voltage": voltage,
    "Current": current,
    "Temperature": temperature,
    "Vibration": vibration,
    "Speed": speed,
    "Power": power
})

# ==========================================
# 5. DISPLAY AND SAVE DATA
# ==========================================

print("\nHEALTHY MOTOR SIMULATION RESULTS\n")

print(data.head(10).round(2))

data.to_csv("data/healthy_motor_data.csv", index=False)

print("\nDataset saved successfully!")

# ==========================================
# 6. VISUALIZATION
# ==========================================

parameters = [
    "Voltage",
    "Current",
    "Temperature",
    "Vibration",
    "Speed",
    "Power"
]

fig, axes = plt.subplots(3, 2, figsize=(13, 10))

for ax, parameter in zip(axes.flatten(), parameters):

    ax.plot(data["Time"], data[parameter])

    ax.set_title(parameter + " vs Time")
    ax.set_xlabel("Time")
    ax.set_ylabel(parameter)
    ax.grid(True)

plt.suptitle(
    "Industrial Motor - Healthy Operating Condition",
    fontsize=15
)

plt.tight_layout()
plt.show()