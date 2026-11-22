import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATASETS
# ==========================================

healthy = pd.read_csv("data/healthy_motor_data.csv")
faulty = pd.read_csv("data/faulty_motor_data.csv")

print("Datasets loaded successfully!")

# ==========================================
# 2. SEPARATE FAULT CONDITIONS
# ==========================================

overload = faulty[faulty["Condition"] == "Overload"]
bearing = faulty[faulty["Condition"] == "Bearing Deterioration"]
cooling = faulty[faulty["Condition"] == "Cooling Failure"]
voltage_fault = faulty[faulty["Condition"] == "Voltage Variation"]
mechanical = faulty[faulty["Condition"] == "Mechanical Loading"]

# ==========================================
# 3. GRAPH CONFIGURATION
# ==========================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

fig, axes = plt.subplots(
    3, 2,
    figsize=(18, 14)
)

fig.suptitle(
    "INDUSTRIAL MOTOR - HEALTHY VS FAULTY CONDITIONS",
    fontsize=19,
    fontweight="bold",
    y=0.98
)

# ==========================================
# 4. CURRENT COMPARISON
# ==========================================

axes[0, 0].plot(
    healthy["Time"],
    healthy["Current"],
    label="Healthy",
    linewidth=2
)

axes[0, 0].plot(
    overload["Time"],
    overload["Current"],
    label="Overload",
    linewidth=2
)

axes[0, 0].set_title("Current Comparison")
axes[0, 0].set_xlabel("Time (seconds)")
axes[0, 0].set_ylabel("Current (A)")
axes[0, 0].legend(loc="upper left")
axes[0, 0].grid(alpha=0.3)

# ==========================================
# 5. VIBRATION COMPARISON
# ==========================================

axes[0, 1].plot(
    healthy["Time"],
    healthy["Vibration"],
    label="Healthy",
    linewidth=2
)

axes[0, 1].plot(
    bearing["Time"],
    bearing["Vibration"],
    label="Bearing Deterioration",
    linewidth=2
)

axes[0, 1].set_title("Vibration Comparison")
axes[0, 1].set_xlabel("Time (seconds)")
axes[0, 1].set_ylabel("Vibration (mm/s)")
axes[0, 1].legend(loc="upper left")
axes[0, 1].grid(alpha=0.3)

# ==========================================
# 6. TEMPERATURE COMPARISON
# ==========================================

axes[1, 0].plot(
    healthy["Time"],
    healthy["Temperature"],
    label="Healthy",
    linewidth=2
)

axes[1, 0].plot(
    cooling["Time"],
    cooling["Temperature"],
    label="Cooling Failure",
    linewidth=2
)

axes[1, 0].set_title("Temperature Comparison")
axes[1, 0].set_xlabel("Time (seconds)")
axes[1, 0].set_ylabel("Temperature (°C)")
axes[1, 0].legend(loc="upper left")
axes[1, 0].grid(alpha=0.3)

# ==========================================
# 7. VOLTAGE COMPARISON
# ==========================================

axes[1, 1].plot(
    healthy["Time"],
    healthy["Voltage"],
    label="Healthy",
    linewidth=2
)

axes[1, 1].plot(
    voltage_fault["Time"],
    voltage_fault["Voltage"],
    label="Voltage Variation",
    linewidth=2
)

axes[1, 1].set_title("Voltage Comparison")
axes[1, 1].set_xlabel("Time (seconds)")
axes[1, 1].set_ylabel("Voltage (V)")
axes[1, 1].legend(loc="upper left")
axes[1, 1].grid(alpha=0.3)

# ==========================================
# 8. SPEED COMPARISON
# ==========================================

axes[2, 0].plot(
    healthy["Time"],
    healthy["Speed"],
    label="Healthy",
    linewidth=2
)

axes[2, 0].plot(
    mechanical["Time"],
    mechanical["Speed"],
    label="Mechanical Loading",
    linewidth=2
)

axes[2, 0].set_title("Speed Comparison")
axes[2, 0].set_xlabel("Time (seconds)")
axes[2, 0].set_ylabel("Speed (RPM)")
axes[2, 0].legend(loc="upper left")
axes[2, 0].grid(alpha=0.3)

# ==========================================
# 9. POWER COMPARISON
# ==========================================

axes[2, 1].plot(
    healthy["Time"],
    healthy["Power"],
    label="Healthy",
    linewidth=2
)

datasets = [
    ("Overload", overload),
    ("Bearing", bearing),
    ("Cooling Failure", cooling),
    ("Voltage Variation", voltage_fault),
    ("Mechanical Loading", mechanical)
]

for name, dataset in datasets:

    axes[2, 1].plot(
        dataset["Time"],
        dataset["Power"],
        label=name,
        linewidth=1.8
    )

axes[2, 1].set_title("Electrical Power Comparison")
axes[2, 1].set_xlabel("Time (seconds)")
axes[2, 1].set_ylabel("Power (kW)")

axes[2, 1].legend(
    loc="upper left",
    fontsize=8
)

axes[2, 1].grid(alpha=0.3)

# ==========================================
# 10. FINAL LAYOUT
# ==========================================

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save high-resolution image

plt.savefig(
    "outputs/motor_condition_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nVisualization completed successfully!")
print("High-resolution graph saved successfully!")
