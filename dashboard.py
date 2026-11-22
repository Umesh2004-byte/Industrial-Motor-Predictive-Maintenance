import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# --------------------------------------------------
# STEP 1: PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Industrial Motor Monitoring",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ INDUSTRIAL MOTOR MONITORING SYSTEM")

st.caption(
    "Predictive Maintenance | Anomaly Detection | "
    "Degradation Monitoring"
)

st.info(
    "Demonstration dashboard using simulated industrial motor data. "
    "Values are not live measurements."
)


# --------------------------------------------------
# STEP 2: LOAD DATASETS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

@st.cache_data
def load_data():

    healthy = pd.read_csv(
        DATA_DIR / "healthy_motor_data.csv"
    )

    faulty = pd.read_csv(
        DATA_DIR / "faulty_motor_data.csv"
    )

    anomaly = pd.read_csv(
        DATA_DIR / "anomaly_detection_results.csv"
    )

    degradation = pd.read_csv(
        DATA_DIR / "bearing_degradation_analysis.csv"
    )

    return healthy, faulty, anomaly, degradation


try:

    healthy, faulty, anomaly, degradation = load_data()

    st.success("All datasets loaded successfully!")

except FileNotFoundError as error:

    st.error(f"Dataset not found: {error}")
    st.stop()


# --------------------------------------------------
# STEP 3: SIDEBAR
# --------------------------------------------------

st.sidebar.title("Motor Monitoring")

page = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Motor Overview",
        "Fault Analysis",
        "Anomaly Detection",
        "Degradation Monitoring"
    ]
)


# --------------------------------------------------
# STEP 4: MOTOR OVERVIEW
# --------------------------------------------------

if page == "Motor Overview":

    st.header("Motor Operating Parameters")

    st.write(
        "Select a motor operating condition "
        "to inspect its parameters."
    )

    condition = st.selectbox(
        "Select Motor Condition",
        ["Healthy"] + faulty["Condition"].unique().tolist()
    )

    if condition == "Healthy":

        selected_data = healthy

    else:

        selected_data = faulty[
            faulty["Condition"] == condition
        ]

    latest = selected_data.iloc[-1]

    # Motor parameter cards

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Voltage",
        f"{latest['Voltage']:.2f} V"
    )

    col2.metric(
        "Current",
        f"{latest['Current']:.2f} A"
    )

    col3.metric(
        "Temperature",
        f"{latest['Temperature']:.2f} °C"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Vibration",
        f"{latest['Vibration']:.2f} mm/s"
    )

    col5.metric(
        "Speed",
        f"{latest['Speed']:.2f} RPM"
    )

    col6.metric(
        "Power",
        f"{latest['Power']:.2f} kW"
    )

    st.subheader("Motor Parameter Trends")

    parameter = st.selectbox(
        "Choose Parameter",
        [
            "Voltage",
            "Current",
            "Temperature",
            "Vibration",
            "Speed",
            "Power"
        ]
    )

    fig = px.line(
        selected_data,
        x="Time",
        y=parameter,
        title=f"{parameter} vs Time",
        markers=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Motor Dataset")

    st.dataframe(
        selected_data,
        use_container_width=True
    )


# --------------------------------------------------
# STEP 5: FAULT ANALYSIS
# --------------------------------------------------

elif page == "Fault Analysis":

    st.header("Motor Fault Analysis")

    fault_summary = faulty.groupby(
        "Condition"
    ).size().reset_index(name="Observations")

    fig = px.bar(
        fault_summary,
        x="Condition",
        y="Observations",
        title="Simulated Fault Dataset Distribution",
        text="Observations"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Healthy vs Faulty Comparison")

    parameter = st.selectbox(
        "Select Motor Parameter",
        [
            "Voltage",
            "Current",
            "Temperature",
            "Vibration",
            "Speed",
            "Power"
        ]
    )

    healthy_plot = healthy[
        ["Time", parameter]
    ].copy()

    healthy_plot["Condition"] = "Healthy"

    faulty_plot = faulty[
        ["Time", parameter, "Condition"]
    ].copy()

    comparison = pd.concat(
        [healthy_plot, faulty_plot],
        ignore_index=True
    )

    fig = px.line(
        comparison,
        x="Time",
        y=parameter,
        color="Condition",
        title=f"{parameter} Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# --------------------------------------------------
# STEP 6: ANOMALY DETECTION
# --------------------------------------------------

elif page == "Anomaly Detection":

    st.header("Machine Learning - Anomaly Detection")

    normal_count = (
        anomaly["Predicted_Label"] == 0
    ).sum()

    anomaly_count = (
        anomaly["Predicted_Label"] == 1
    ).sum()

    total = len(anomaly)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Observations",
        total
    )

    col2.metric(
        "Normal Observations",
        int(normal_count)
    )

    col3.metric(
        "Anomalies Detected",
        int(anomaly_count)
    )

    st.subheader("Anomaly Detection Distribution")

    summary = anomaly[
        "Detection_Status"
    ].value_counts().reset_index()

    summary.columns = [
        "Status",
        "Count"
    ]

    fig = px.pie(
        summary,
        names="Status",
        values="Count",
        title="Normal vs Anomaly Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Fault Detection Performance")

    condition_summary = anomaly.groupby(
        ["Condition", "Detection_Status"],
        dropna=False
    ).size().reset_index(name="Count")

    condition_summary["Condition"] = (
        condition_summary["Condition"]
        .fillna("Healthy")
    )

    fig = px.bar(
        condition_summary,
        x="Condition",
        y="Count",
        color="Detection_Status",
        barmode="group",
        title="Anomaly Detection by Motor Condition"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Prediction Results")

    st.dataframe(
        anomaly,
        use_container_width=True
    )


# --------------------------------------------------
# STEP 7: DEGRADATION MONITORING
# --------------------------------------------------

elif page == "Degradation Monitoring":

    st.header("Bearing Degradation Monitoring")

    latest_index = degradation[
        "Rolling_Degradation"
    ].iloc[-1]

    maximum_index = degradation[
        "Rolling_Degradation"
    ].max()

    col1, col2 = st.columns(2)

    col1.metric(
        "Current Degradation Index",
        f"{latest_index:.2f}"
    )

    col2.metric(
        "Maximum Degradation Index",
        f"{maximum_index:.2f}"
    )

    WARNING_THRESHOLD = 3.0

    if latest_index >= WARNING_THRESHOLD:

        st.error(
            "WARNING: Degradation index has exceeded "
            "the illustrative warning threshold!"
        )

    else:

        st.success(
            "Motor degradation index is below "
            "the illustrative warning threshold."
        )

    st.subheader("Degradation Trend")

    fig = px.line(
        degradation,
        x="Time",
        y=[
            "Degradation_Index",
            "Rolling_Degradation",
            "Predicted_Degradation"
        ],
        title="Bearing Degradation Trend"
    )

    fig.add_hline(
        y=WARNING_THRESHOLD,
        line_dash="dash",
        annotation_text="Warning Threshold"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Degradation Dataset")

    st.dataframe(
        degradation,
        use_container_width=True
    )


# --------------------------------------------------
# STEP 8: FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Industrial Motor Predictive Maintenance System | "
    "Simulation-Based Demonstration"
)