# ui/app.py
import streamlit as st
import requests
import pandas as pd
import altair as alt

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Accelerator Optimization Copilot", layout="wide")

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title(" Accelerator Optimization Copilot")
st.caption("Model Workload → ML Scheduling → Cache Simulation → Compiler Prediction + Validation")


# ---------------------------------------------------------
# Layout Inputs
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    model_type = st.selectbox(
        "Select Model Workload",
        ["Transformer", "CNN", "MLP", "MoE"]
    )
    policy = st.selectbox(
        "Cache Policy",
        ["LRU", "FIFO", "ML"]
    )

with col2:
    num_layers = st.slider("Layers", 4, 48, 12)
    batch_size = st.slider("Batch Size", 1, 256, 32)
    seq_length = st.slider("Sequence Length", 32, 1024, 128)
    reuse_probability = st.slider("Reuse Probability", 0.0, 1.0, 0.3)

run_validation = st.checkbox("Run Statistical Validation", value=True)

# ---------------------------------------------------------
# RUN SIMULATION
# ---------------------------------------------------------

if st.button(" Run Simulation"):
    with st.spinner("Generating workload and running simulation..."):

        payload = {
            "model_type": model_type,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "reuse_probability": reuse_probability
        }

        # Simulation metrics
        try:
            sim_res = requests.post(
                f"{API_URL}/simulate?policy={policy}",
                json=payload
            ).json()

            st.subheader(" Simulation Metrics")
            st.json(sim_res)
        except Exception as e:
            st.error(f"Error running simulation: {e}")

        # ML eviction schedule
        try:
            schedule_res = requests.post(
                f"{API_URL}/predict-schedule",
                json=payload
            ).json()

            df = pd.DataFrame(schedule_res)
            
            # ---------------------------------------------------------
            # Side-by-side layout: Eviction Scores + Heatmap
            # ---------------------------------------------------------
            
            col_scores, col_heatmap = st.columns(2)
            
            # drop 'op' column if exists, as per user request
            if "op" in df.columns:
                df = df.drop(columns=["op"])

            with col_scores:
                st.subheader(" Predicted Eviction Scores (Top 20)")
                st.dataframe(df.head(20), height=400)
            
            with col_heatmap:
                if {"tensor_id", "reuse_distance", "eviction_score"}.issubset(df.columns):
                    st.subheader(" Cache Occupancy Heatmap")

                    heatmap = (
                        alt.Chart(df.head(50))
                        .mark_rect()
                        .encode(
                            x="tensor_id:O",
                            y="reuse_distance:Q",
                            color="eviction_score:Q",
                            tooltip=["tensor_id", "reuse_distance", "eviction_score"]
                        )
                        .properties(height=400)
                    )

                    st.altair_chart(heatmap, use_container_width=True)
        except Exception as e:
            st.error(f"Error predicting schedule: {e}")

        # ---------------------------------------------------------
        # Statistical Validation
        # ---------------------------------------------------------

        if run_validation:
            st.markdown("---")
            st.header(" Statistical Validation")

            try:
                val_res = requests.post(
                    f"{API_URL}/validate-trace",
                    json=payload
                ).json()

                st.subheader("Validation Results (p-values)")
                st.json(val_res)
            except Exception as e:
                st.error(f"Error running validation: {e}")


# ---------------------------------------------------------
# COMPILER PREDICTION (from Code 2)
# ---------------------------------------------------------

st.markdown("---")
st.header("Compiler Prediction (Latency + Energy)")

code_input = st.text_area("Paste CUDA / PyTorch / CPU Kernel Code Here:")

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    pred_model_type = st.selectbox("Model Type", ["Transformer", "CNN", "MLP", "MoE"], key="pred_model")
with colB:
    opt_level = st.selectbox("Optimization Level", ["O0", "O1", "O2", "O3"])
with colC:
    run_pred = st.button("Predict Compilation Results")

if run_pred and code_input.strip():
    with st.spinner("Running compiler prediction..."):
        try:
            response = requests.post(
                f"{API_URL}/predict-compile",
                json={"code": code_input, "model_type": pred_model_type, "opt_level": opt_level}
            ).json()

            st.subheader("Latency / Energy Prediction")
            st.json(response)

            # Feature importance bar chart
            fi = pd.DataFrame(
                {
                    "feature": list(response["feature_importance"].keys()),
                    "importance": list(response["feature_importance"].values())
                }
            )

            st.subheader(" Feature Importance")

            fi_chart = alt.Chart(fi).mark_bar().encode(
                x="importance:Q",
                y=alt.Y("feature:N", sort="-x")
            )

            st.altair_chart(fi_chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error running compiler prediction: {e}")
