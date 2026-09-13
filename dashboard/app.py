from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# PAGE CONFIG

st.set_page_config(
    page_title="GlycemicPlus",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)


# STYLING

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.5rem 1.8rem;
            border-radius: 18px;
            background: linear-gradient(
                135deg,
                rgba(220,38,38,0.12),
                rgba(59,130,246,0.08)
            );
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.3rem;
        }

        .hero p {
            margin-top: 0.4rem;
            opacity: 0.8;
        }

        .state-card {
            padding: 1.25rem;
            border-radius: 16px;
            border: 1px solid rgba(128,128,128,0.25);
            background: rgba(128,128,128,0.04);
        }

        .small-label {
            font-size: 0.8rem;
            opacity: 0.65;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .big-value {
            font-size: 2rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }

        .disclaimer {
            font-size: 0.8rem;
            opacity: 0.7;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# PATHS


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

REPLAY_FILE = DATA_DIR / "glycemic_sentinel_replay.json"
MODEL_FILE = DATA_DIR / "personalized_model_comparison.csv"



# LOAD DATA

@st.cache_data
def load_replay():

    df = pd.read_json(REPLAY_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["correct"] = df["correct"].astype(bool)

    return df


@st.cache_data
def load_models():

    return pd.read_csv(MODEL_FILE)


df = load_replay()
models = load_models()

# UNCERTAINTY

CONFIDENCE_THRESHOLD = 0.45
MARGIN_THRESHOLD = 0.05


def prediction_state(row):

    probs = np.array([
        row["p_hypo"],
        row["p_normal"],
        row["p_hyper"]
    ])

    ordered = np.sort(probs)

    margin = (
        ordered[-1]
        -
        ordered[-2]
    )

    confidence = probs.max()

    if (
        confidence < CONFIDENCE_THRESHOLD
        or margin < MARGIN_THRESHOLD
    ):
        return "Uncertain", confidence, margin

    return (
        row["predicted_label"],
        confidence,
        margin
    )


# HERO

st.markdown(
    """
    <div class="hero">
        <h1>GlycemicPlus</h1>
        <p>
            Wearable PPG-based glycemic-state estimation using
            gradient-boosted physiological features and deep temporal learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# SIDEBAR

st.sidebar.title("Replay Controls")

subjects = sorted(
    df["subject"].unique()
)

subject = st.sidebar.selectbox(
    "Participant",
    subjects
)

subject_df = (
    df[df["subject"] == subject]
    .sort_values("timestamp")
    .reset_index(drop=True)
)

index = st.sidebar.slider(
    "Observation",
    min_value=0,
    max_value=len(subject_df) - 1,
    value=0,
    step=1
)

show_truth = st.sidebar.toggle(
    "Reveal Dexcom ground truth",
    value=False
)

st.sidebar.divider()

st.sidebar.caption(
    f"{len(subject_df):,} replay observations for {subject}"
)

st.sidebar.caption(
    "Prediction output is based on PPG-derived model probabilities. "
    "Dexcom is used only for validation."
)


row = subject_df.iloc[index]

display_state, confidence, margin = prediction_state(row)

history = subject_df[
    subject_df["timestamp"] <= row["timestamp"]
].copy()


# TABS

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Live Replay",
        "Model Performance",
        "Error Analysis",
        "Research Overview"
    ]
)


# TAB 1 — REPLAY

with tab1:

    st.subheader("Current Observation")

    st.caption(
        f"{subject} • "
        f"{row['timestamp'].strftime('%b %d, %Y %I:%M %p')}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="state-card">
                <div class="small-label">Model State</div>
                <div class="big-value">{display_state}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.1f}%"
        )

    with col3:

        st.metric(
            "Top-two margin",
            f"{margin * 100:.1f}%"
        )

    with col4:

        if show_truth:

            st.metric(
                "Dexcom",
                f"{row['glucose']:.0f} mg/dL"
            )

        else:

            st.metric(
                "Dexcom",
                "Hidden"
            )


    st.markdown("### Prediction Probabilities")

    probability_df = pd.DataFrame({
        "State": [
            "Hypoglycemia",
            "In Range",
            "Hyperglycemia"
        ],
        "Probability": [
            row["p_hypo"] * 100,
            row["p_normal"] * 100,
            row["p_hyper"] * 100
        ]
    })

    prob_fig = px.bar(
        probability_df,
        x="Probability",
        y="State",
        orientation="h",
        text="Probability"
    )

    prob_fig.update_traces(
        texttemplate="%{text:.1f}%"
    )

    prob_fig.update_layout(
        xaxis_title="Probability (%)",
        yaxis_title="",
        xaxis_range=[0, 100],
        height=280,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(
        prob_fig,
        use_container_width=True
    )


    if show_truth:

        left, right = st.columns(2)

        with left:

            st.markdown("### Dexcom Ground Truth")

            st.metric(
                "Actual glucose",
                f"{row['glucose']:.0f} mg/dL"
            )

            st.metric(
                "Actual state",
                row["actual_label"]
            )

        with right:

            st.markdown("### Validation Result")

            if row["correct"]:

                st.success(
                    "Raw model classification was correct."
                )

            else:

                st.error(
                    "Raw model classification was incorrect."
                )

            st.write(
                f"Raw model output: **{row['predicted_label']}**"
            )

            st.write(
                f"Application display state: **{display_state}**"
            )


    st.divider()

    st.subheader("Timeline")

    chart1, chart2 = st.columns(2)

    with chart1:

        glucose_fig = go.Figure()

        glucose_fig.add_trace(
            go.Scatter(
                x=history["timestamp"],
                y=history["glucose"],
                mode="lines",
                name="Dexcom glucose"
            )
        )

        glucose_fig.add_hline(
            y=70,
            line_dash="dash",
            annotation_text="70"
        )

        glucose_fig.add_hline(
            y=180,
            line_dash="dash",
            annotation_text="180"
        )

        glucose_fig.update_layout(
            title="Dexcom glucose trajectory",
            xaxis_title="Time",
            yaxis_title="Glucose (mg/dL)",
            height=360
        )

        st.plotly_chart(
            glucose_fig,
            use_container_width=True
        )

    with chart2:

        probability_fig = go.Figure()

        probability_fig.add_trace(
            go.Scatter(
                x=history["timestamp"],
                y=history["p_hypo"],
                mode="lines",
                name="Hypoglycemia"
            )
        )

        probability_fig.add_trace(
            go.Scatter(
                x=history["timestamp"],
                y=history["p_normal"],
                mode="lines",
                name="In Range"
            )
        )

        probability_fig.add_trace(
            go.Scatter(
                x=history["timestamp"],
                y=history["p_hyper"],
                mode="lines",
                name="Hyperglycemia"
            )
        )

        probability_fig.update_layout(
            title="Model probability trajectory",
            xaxis_title="Time",
            yaxis_title="Probability",
            yaxis_range=[0, 1],
            height=360
        )

        st.plotly_chart(
            probability_fig,
            use_container_width=True
        )


# TAB 2 — PERFORMANCE

with tab2:

    st.subheader("Model Comparison")

    model_long = models.melt(
        id_vars="model",
        value_vars=[
            "macro_f1",
            "balanced_accuracy"
        ],
        var_name="Metric",
        value_name="Score"
    )

    model_long["Metric"] = model_long["Metric"].replace({
        "macro_f1": "Macro F1",
        "balanced_accuracy": "Balanced Accuracy"
    })

    model_fig = px.bar(
        model_long,
        x="model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=".3f"
    )

    model_fig.update_layout(
        yaxis_range=[0, 0.5],
        xaxis_title="Model",
        yaxis_title="Score",
        height=400
    )

    st.plotly_chart(
        model_fig,
        use_container_width=True
    )


    hybrid = models[
        models["model"] == "Hybrid"
    ].iloc[0]

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Hybrid Macro F1",
        f"{hybrid['macro_f1']:.4f}"
    )

    m2.metric(
        "Balanced Accuracy",
        f"{hybrid['balanced_accuracy']:.4f}"
    )

    m3.metric(
        "Replay Observations",
        f"{len(df):,}"
    )

    m4.metric(
        "Participants",
        df["subject"].nunique()
    )


    st.divider()

    st.subheader("Confusion Matrix")

    confusion = pd.crosstab(
        df["actual_label"],
        df["predicted_label"],
        normalize="index"
    )

    confusion_long = (
        confusion
        .reset_index()
        .melt(
            id_vars="actual_label",
            var_name="Predicted",
            value_name="Rate"
        )
    )

    confusion_fig = px.density_heatmap(
        confusion_long,
        x="Predicted",
        y="actual_label",
        z="Rate",
        text_auto=".2f",
        histfunc="avg"
    )

    confusion_fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=420
    )

    st.plotly_chart(
        confusion_fig,
        use_container_width=True
    )


# TAB 3 — ERROR ANALYSIS
with tab3:

    st.subheader("Where the Model Fails")

    errors = df[
        ~df["correct"]
    ].copy()

    error_rate = (
        len(errors)
        /
        len(df)
    )

    e1, e2, e3 = st.columns(3)

    e1.metric(
        "Incorrect predictions",
        f"{len(errors):,}"
    )

    e2.metric(
        "Error rate",
        f"{error_rate * 100:.1f}%"
    )

    e3.metric(
        "Average error confidence",
        f"{errors['confidence'].mean() * 100:.1f}%"
        if "confidence" in errors.columns
        else "N/A"
    )


    st.markdown("### Most Confident Errors")

    if "confidence" not in errors.columns:

        errors["confidence"] = errors[
            [
                "p_hypo",
                "p_normal",
                "p_hyper"
            ]
        ].max(axis=1)

    confident_errors = (
        errors
        .sort_values(
            "confidence",
            ascending=False
        )
        .head(25)
    )

    st.dataframe(
        confident_errors[
            [
                "subject",
                "timestamp",
                "glucose",
                "actual_label",
                "predicted_label",
                "confidence",
                "p_hypo",
                "p_normal",
                "p_hyper"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


    st.markdown("### Prediction Error Patterns")

    error_matrix = pd.crosstab(
        errors["actual_label"],
        errors["predicted_label"]
    )

    st.dataframe(
        error_matrix,
        use_container_width=True
    )


# TAB 4 — OVERVIEW
with tab4:

    st.subheader("How GlycemicPlus Works")

    st.markdown(
        """
        ### 1. Wearable signal

        GlycemicPlus analyzes five-minute PPG waveform windows.

        ### 2. Two complementary models

        **CatBoost**
        uses engineered physiological and frequency-domain features.

        **Temporal Convolutional Network**
        learns temporal representations directly from raw PPG.

        ### 3. Hybrid prediction

        Validation-selected probability fusion combines both models.

        ### 4. Ground-truth validation

        Dexcom CGM values are used only after prediction to evaluate
        whether the model classified the glycemic state correctly.

        ### Evaluation protocol

        Each participant is split chronologically:

        **70% train → 15% validation → 15% test**

        This represents a personalized setting where earlier data
        calibrates the model and later observations are evaluated.

        ### Current limitations

        The current model shows only modest discriminative performance.
        It should be treated as a feasibility study, not a clinical tool.

        Future work includes multimodal wearable signals, Conformer-based
        representation learning, larger cohorts, and prospective validation.
        """
    )


# FOOTER
st.markdown(
    """
    <div class="disclaimer">
    GlycemicPlus is an experimental research prototype.
    It is not intended to diagnose, treat, or guide insulin dosing,
    and is not a replacement for continuous glucose monitoring.
    </div>
    """,
    unsafe_allow_html=True
)
