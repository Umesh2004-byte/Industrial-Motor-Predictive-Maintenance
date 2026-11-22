import numpy as np
import pandas as pd

# ==========================================
# 1. MOTOR SPECIFICATIONS
# ==========================================

RATED_VOLTAGE = 415
RATED_CURRENT = 6
RATED_SPEED = 1440
POWER_FACTOR = 0.85

np.random.seed(42)

# ==========================================
# 2. LOAD HEALTHY MOTOR DATA
# ==========================================

healthy_data = pd.read_csv("data/healthy_motor_data.csv")

print("\nHealthy motor dataset loaded successfully!")

# ==========================================
# 3. OVERLOAD CONDITION
# ==========================================

overload = healthy_data.copy()

overload["Current"] = (
    healthy_data["Current"] * 1.35
)

overload["Temperature"] = (
    healthy_data["Temperature"] + np.linspace(0, 20, 100)
)

overload["Vibration"] = (
    healthy_data["Vibration"] + np.linspace(0, 0.8, 100)
)

overload["Speed"] = (
    healthy_data["Speed"] - np.linspace(0, 30, 100)
)

overload["Condition"] = "Overload"
overload["Label"] = 1

# ==========================================
# 4. BEARING DETERIORATION
# ==========================================

bearing = healthy_data.copy()

bearing["Vibration"] = (
    healthy_data["Vibration"] + np.linspace(0, 3, 100)
)

bearing["Temperature"] = (
    healthy_data["Temperature"] + np.linspace(0, 12, 100)
)

bearing["Current"] = (
    healthy_data["Current"] * 1.08
)

bearing["Condition"] = "Bearing Deterioration"
bearing["Label"] = 1

# ==========================================
# 5. COOLING FAILURE
# ==========================================

cooling = healthy_data.copy()

cooling["Temperature"] = (
    healthy_data["Temperature"] + np.linspace(0, 30, 100)
)

cooling["Condition"] = "Cooling Failure"
cooling["Label"] = 1

# ==========================================
# 6. VOLTAGE VARIATION
# ==========================================

voltage_fault = healthy_data.copy()

voltage_fault["Voltage"] = (
    healthy_data["Voltage"] - np.linspace(0, 35, 100)
)

voltage_fault["Current"] = (
    healthy_data["Current"] * 1.12
)

voltage_fault["Speed"] = (
    healthy_data["Speed"] - np.linspace(0, 15, 100)
)

voltage_fault["Condition"] = "Voltage Variation"
voltage_fault["Label"] = 1

# ==========================================
# 7. MECHANICAL LOADING
# ==========================================

mechanical = healthy_data.copy()

mechanical["Current"] = (
    healthy_data["Current"] * 1.25
)

mechanical["Speed"] = (
    healthy_data["Speed"] - np.linspace(0, 45, 100)
)

mechanical["Temperature"] = (
    healthy_data["Temperature"] + np.linspace(0, 15, 100)
)

mechanical["Vibration"] = (
    healthy_data["Vibration"] + np.linspace(0, 1.5, 100)
)

mechanical["Condition"] = "Mechanical Loading"
mechanical["Label"] = 1

# ==========================================
# 8. RECALCULATE ELECTRICAL POWER
# ==========================================

fault_datasets = [
    overload,
    bearing,
    cooling,
    voltage_fault,
    mechanical
]

for dataset in fault_datasets:

    dataset["Power"] = (
        np.sqrt(3)
        * dataset["Voltage"]
        * dataset["Current"]
        * POWER_FACTOR
        / 1000
    )

# ==========================================
# 9. COMBINE ALL FAULT CONDITIONS
# ==========================================

fault_data = pd.concat(
    fault_datasets,
    ignore_index=True
)

fault_data.to_csv(
    "data/faulty_motor_data.csv",
    index=False
)

# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

print("\nFAULT SIMULATION COMPLETED!")

print("\nFault Dataset Preview:\n")

print(fault_data.head(10).round(2))

print("\nFault Condition Summary:\n")

print(fault_data.groupby("Condition").size())

print("\nTotal Fault Observations:", len(fault_data))

print("\nDataset saved successfully!")
