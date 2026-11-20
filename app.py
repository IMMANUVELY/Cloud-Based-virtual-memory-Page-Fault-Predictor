
"""
Streamlit Dashboard
Cloud Based Virtual Memory Page Fault Predictor
NO AI VERSION

Run:
    streamlit run app.py

Requires:
    main_no_ai.py
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
import streamlit as st

from main_no_ai import (
    generate_trace,
    simulate_lru,
    features_from_history,
    predict,
    calculate_metrics,
    WINDOW
)


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Virtual Memory Predictor",
    page_icon="💻",
    layout="wide"
)


# DASHBOARD TITLE
st.title("💻 Cloud Based Virtual Memory Page Fault Predictor")

st.caption(
    "Computer Architecture Project | "
    "Rule-Based Prediction | No AI / No Machine Learning"
)

st.divider()


# SIDEBAR SETTINGS
st.sidebar.title("⚙️ Settings")

source = st.sidebar.radio(
    "Select Page Reference String",
    ["Generated Trace", "Type My Own"]
)

frames = st.sidebar.slider(
    "RAM Frames",
    min_value=2,
    max_value=32,
    value=16
)

threshold = st.sidebar.slider(
    "Prediction Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

use_cloud = st.sidebar.checkbox(
    "Use Cloud API (Flask)"
)

api_url = st.sidebar.text_input(
    "API URL",
    "http://127.0.0.1:5000/predict",
    disabled=not use_cloud
)


# INPUT MEMORY TRACE
if source == "Generated Trace":

    st.sidebar.subheader("Trace Settings")

    n = st.sidebar.slider(
        "Number of Memory Accesses",
        100,
        2000,
        300,
        50
    )

    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=7
    )

    trace = generate_trace(
        n=n,
        seed=int(seed)
    )

else:

    st.sidebar.subheader("Custom Memory Pages")

    page_input = st.sidebar.text_area(
        "Enter page numbers separated by spaces",
        value="1 2 3 4 1 2 5 1 2 3 4 5 1 2 3 4 5 6 7 8 1 2 3"
    )

    try:
        trace = [
            int(x) for x in page_input.split()
        ]

    except ValueError:
        st.error("Please enter valid integer page numbers.")
        st.stop()


# VALIDATE INPUT
if len(trace) < WINDOW + 2:

    st.warning(
        f"Please enter at least {WINDOW + 2} memory accesses."
    )
    st.stop()


if frames < 1:
    st.error("RAM frames must be greater than zero.")
    st.stop()


# SIMULATION BUTTON
run_simulation = st.sidebar.button(
    "▶ Run Simulation",
    type="primary",
    use_container_width=True
)

if "run_simulation" not in st.session_state:
    st.session_state.run_simulation = True

if run_simulation:
    st.session_state.run_simulation = True

if not st.session_state.run_simulation:
    st.info("Click Run Simulation to start.")
    st.stop()


# STEP 1: RAM SIMULATION
faults = simulate_lru(
    trace,
    frames=frames
)

ram = OrderedDict()
rows = []

for t, page in enumerate(trace):

    if page in ram:

        ram.move_to_end(page)
        result = "HIT"

    else:

        result = "FAULT"

        if len(ram) >= frames:
            ram.popitem(last=False)

        ram[page] = True

    rows.append({
        "Step": t + 1,
        "Page": page,
        "Result": result,
        "RAM Contents": " ".join(
            map(str, ram.keys())
        )
    })


sim_df = pd.DataFrame(rows)


# STEP 2: PREDICTION
def get_probability(features):

    if use_cloud:

        import requests

        response = requests.post(
            api_url,
            json={"features": features},
            timeout=5
        )

        response.raise_for_status()

        return response.json()["fault_probability"]

    else:

        return predict(
            features,
            threshold=threshold
        )["fault_probability"]


predictions = []

try:

    for t in range(WINDOW - 1, len(trace) - 1):

        features = features_from_history(
            trace[t - WINDOW + 1:t + 1],
            faults[t - WINDOW + 1:t + 1]
        )

        probability = get_probability(features)

        action = (
            "PREFETCH"
            if probability > threshold
            else "NORMAL"
        )

        predictions.append({

            "Step": t + 1,

            "Current Page": trace[t],

            "Fault Probability": probability,

            "Action": action,

            "Predicted Next Fault": int(
                probability > threshold
            ),

            "Actual Next Fault": faults[t + 1]

        })

except Exception as e:

    st.error(f"Prediction error: {e}")
    st.info(
        "If Cloud API is enabled, make sure the Flask server is running."
    )
    st.stop()


pred_df = pd.DataFrame(predictions)

if pred_df.empty:
    st.error("No predictions available.")
    st.stop()


# STEP 3: METRICS
actual = pred_df["Actual Next Fault"].to_numpy()
predicted = pred_df["Predicted Next Fault"].to_numpy()

metrics = calculate_metrics(
    actual,
    predicted
)

total_accesses = len(trace)
total_faults = sum(faults)
total_hits = total_accesses - total_faults
fault_rate = np.mean(faults)

prefetch_count = int(
    (pred_df["Action"] == "PREFETCH").sum()
)


# DASHBOARD METRIC CARDS
st.subheader("📊 System Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Accesses",
    total_accesses
)

c2.metric(
    "Page Faults",
    total_faults
)

c3.metric(
    "Page Hits",
    total_hits
)

c4.metric(
    "Fault Rate",
    f"{fault_rate:.2%}"
)


st.divider()


# PREDICTION SUMMARY
st.subheader("🔮 Prediction Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Accuracy",
    f"{metrics['accuracy']:.2%}"
)

m2.metric(
    "Precision",
    f"{metrics['precision']:.2%}"
)

m3.metric(
    "Recall",
    f"{metrics['recall']:.2%}"
)

m4.metric(
    "F1 Score",
    f"{metrics['f1_score']:.2%}"
)


st.divider()


# PREDICTION ACTION
st.subheader("⚡ Prediction Actions")

p1, p2 = st.columns(2)

p1.metric(
    "PREFETCH Actions",
    prefetch_count
)

p2.metric(
    "NORMAL Actions",
    len(pred_df) - prefetch_count
)

st.caption(
    "PREFETCH is a predicted recommendation. "
    "This simulation does not physically load pages into RAM."
)


# TABS
tab1, tab2, tab3 = st.tabs([
    "🖥️ RAM Simulation",
    "🔮 Prediction Results",
    "📈 Performance Analysis"
])


# TAB 1: SIMULATION TABLE
with tab1:

    st.subheader("RAM Simulation Table")

    st.write(
        "This table shows each memory access and the "
        "RAM contents after the access using LRU."
    )

    st.dataframe(
        sim_df,
        use_container_width=True,
        height=400
    )

    st.subheader("Memory Access Summary")

    summary = pd.DataFrame({
        "Result": ["Page Hits", "Page Faults"],
        "Count": [total_hits, total_faults]
    })

    st.bar_chart(
        summary.set_index("Result")
    )


# TAB 2: PREDICTIONS
with tab2:

    st.subheader("Next Page Fault Prediction")

    st.write(
        "The following graph displays the predicted "
        "fault probability for each memory access."
    )

    chart_df = pred_df.set_index("Step")[
        ["Fault Probability"]
    ]

    st.line_chart(chart_df)

    st.subheader("Prediction Results Table")

    display_df = pred_df.copy()

    display_df["Fault Probability"] = (
        display_df["Fault Probability"] * 100
    ).round(2).astype(str) + "%"

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )


# TAB 3: PERFORMANCE ANALYSIS
with tab3:

    st.subheader("Actual vs Predicted Page Fault")

    comparison = pred_df.set_index("Step")[
        [
            "Actual Next Fault",
            "Predicted Next Fault"
        ]
    ]

    st.line_chart(comparison)

    st.subheader("Confusion Matrix")

    confusion = pd.DataFrame(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]]
        ],
        index=["Actual Hit", "Actual Fault"],
        columns=["Predicted Hit", "Predicted Fault"]
    )

    st.dataframe(
        confusion,
        use_container_width=True
    )

    st.subheader("Performance Metrics")

    metric_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Value": [
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"]
        ]
    })

    st.bar_chart(
        metric_df.set_index("Metric")
    )


# FOOTER
st.divider()

st.caption(
    "Cloud Based Virtual Memory Page Fault Predictor | "
    "Python + Streamlit + Flask | NO AI VERSION"
)

st.caption(
    "Note: This project uses simulated memory references "
    "and rule-based predictions, not actual operating-system memory."
)
