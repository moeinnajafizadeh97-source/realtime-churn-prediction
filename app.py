import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import shap
import matplotlib.pyplot as plt
from river import forest, drift
from pathlib import Path
from sklearn.metrics import (roc_auc_score, fbeta_score, f1_score,
                             precision_score, recall_score, balanced_accuracy_score)

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

st.markdown("""
<style>
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricLabel"] {
        color: #64748B;
        font-size: 0.85rem;
    }

    /* Section headers — subtle accent underline */
    h2, h3 {
        border-bottom: 2px solid #1E3A5F;
        padding-bottom: 6px;
        color: #1E3A5F;
    }

    /* The primary simulate button — give it real weight */
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #1E3A5F;
        color: white;
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.75em 1.5em;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 6px rgba(30,58,95,0.25);
        transition: transform 0.1s ease;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(30,58,95,0.3);
    }

    /* Alert boxes */
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }

    /* Dataframe container */
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

save_path = str(Path(__file__).parent / "files") + "/"

ARF_THRESHOLD = 0.19
DRIFT_START_BATCH = 5
N_PRETRAIN_BATCHES = 23
ARF_SEED = 2026

DRIFT_STRENGTHS = {5: 0.30, 6: 0.45, 7: 0.60, 8: 0.75, 9: 0.90}

def row_to_dict(row):
    return row.to_dict()

def compute_risk_levels(probs):
    levels = []
    for p in probs:
        if p >= 0.60:
            levels.append("High")
        elif p >= 0.30:
            levels.append("Medium")
        else:
            levels.append("Low")
    return levels


@st.cache_resource
def load_lgbm_model():
    return lgb.Booster(model_file=save_path + 'lgbm_focal_model.txt')

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_data
def load_drifted_batches():
    with open(save_path + 'drifted_batches.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y']

@st.cache_data
def load_train_data():
    X_train = pd.read_csv(save_path + 'X_train.csv')
    y_train = pd.read_csv(save_path + 'y_train.csv').squeeze()
    return X_train, y_train

lgbm_model = load_lgbm_model()
explainer = load_explainer(lgbm_model)
batches_X, batches_y = load_drifted_batches()


if 'initialized' not in st.session_state:
    with st.spinner("Pretraining ARF on historical customer data"):
        X_train, y_train = load_train_data()
        X_train_reset = X_train.reset_index(drop=True)
        y_train_reset = y_train.reset_index(drop=True)
        batch_size = len(X_train_reset) // N_PRETRAIN_BATCHES

        arf_model = forest.ARFClassifier(
            n_models=10, seed=ARF_SEED,
            drift_detector=drift.ADWIN(delta=0.01),
            warning_detector=drift.ADWIN(delta=0.1),
            )

        pretrain_perf = []
        for i in range(N_PRETRAIN_BATCHES):
            start = i * batch_size
            end = start + batch_size if i < N_PRETRAIN_BATCHES - 1 else len(X_train_reset)
            bx = X_train_reset.iloc[start:end].reset_index(drop=True)
            by = y_train_reset.iloc[start:end].reset_index(drop=True)

            batch_probs = []
            for j in range(len(bx)):
                x = row_to_dict(bx.iloc[j])
                y = int(by.iloc[j])
                proba = arf_model.predict_proba_one(x)
                batch_probs.append(proba.get(1, 0.0))
                arf_model.learn_one(x, y)

            batch_probs = np.array(batch_probs)
            y_true = by.values
            y_pred = (batch_probs >= ARF_THRESHOLD).astype(int)
            pretrain_perf.append({
                'Batch': i + 1,
                'AUC': roc_auc_score(y_true, batch_probs) if len(set(y_true)) > 1 else np.nan,
                'F1': f1_score(y_true, y_pred, zero_division=0),
                'F2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'Bal_Acc': balanced_accuracy_score(y_true, y_pred),
            })

    st.session_state.arf_model = arf_model
    st.session_state.step = 0
    st.session_state.history = []
    st.session_state.pretrain_perf = pretrain_perf
    st.session_state.initialized = True


with st.sidebar:
    st.header("About this dashboard")
    st.write("This dashboard demonstrates a churn prediction model that keeps "
             "learning as new customer data arrives.")

    with st.expander("How to use"):
        st.write("Press **Simulate Month** to reveal the next batch of customers. "
                 "The model predicts on the new batch first, then learns from the "
                 "true outcomes before the following batch arrives. This mirrors a "
                 "real deployment, where predictions must be made before the "
                 "outcome is known.\n\n"
                 "Batches 1 to 5 are unchanged. From batch 6 onwards the "
                 "drift starts. Customers on long contracts paying by electronic check "
                 "begin to churn, which is a group of customers the model has learned to "
                 "treat as low risk. The change grows stronger each month, affecting 30% "
                 "of the expected churners in month (batch) 6 and rising to 90% by month 10.")

    with st.expander("The model"):
        st.write("**Adaptive Random Forest**. An ensemble of decision trees for "
                 "streaming data. Each tree monitors its own error rate. When "
                 "accuracy falls, a replacement tree trains in the background and "
                 "takes over if the decline continues.\n\n"
                 f"Customers are flagged as at risk above a probability of "
                 f"{ARF_THRESHOLD}. This threshold was chosen to favour catching "
                 "churners over avoiding false alarms, since a missed churner "
                 "costs more than an unnecessary retention offer.")

    with st.expander("Reading the risk levels"):
        st.write(f"**High** — Churn probability is 60% or higher.\n\n"
                 f"**Medium** — Churn probability is between 30% and 60%.\n\n"
                 f"**Low** — Churn probability is below 30%.\n\n"
                 f"Note that model treats above 19% risk level as churners. "
                 f"So the risk levels introduced here are just for having an understanding on the probability.")


st.title("Live Churn Prediction Simulation - River ARF")
st.info("Click the button below to simulate a new month of customer data arriving. "
        "The model predicts on the new batch first, then learns the true outcomes "
        "before the next batch arrives.")


col_btn, col_reset = st.columns([3, 1])

with col_btn:
    if st.session_state.step < len(batches_X):
        next_batch_num = st.session_state.step + 1
        if st.session_state.step >= DRIFT_START_BATCH:
            if st.session_state.step in DRIFT_STRENGTHS:
                strength = DRIFT_STRENGTHS[st.session_state.step]
            else:
                strength = max(DRIFT_STRENGTHS.values()) if DRIFT_STRENGTHS else 0.0
            label = f"▶️ Simulate Month {next_batch_num} (drift strength {strength:.0%})"
        else:
            label = f"▶️ Simulate Month {next_batch_num} (Batch {next_batch_num})"
        if st.button(label, type="primary", use_container_width=True):
            current_step = st.session_state.step

            if current_step > 0:
                prev_idx = current_step - 1
                bx = batches_X[prev_idx].reset_index(drop=True)
                by = batches_y[prev_idx].reset_index(drop=True)
                for i in range(len(bx)):
                    x = row_to_dict(bx.iloc[i])
                    y = int(by.iloc[i])
                    st.session_state.arf_model.learn_one(x, y)

            new_idx = current_step
            bx = batches_X[new_idx].reset_index(drop=True)
            probs = np.array([
                st.session_state.arf_model.predict_proba_one(row_to_dict(bx.iloc[i])).get(1, 0.0)
                for i in range(len(bx))
            ])

            st.session_state.history.append({'batch_idx': new_idx, 'probs': probs})
            st.session_state.step += 1
            st.rerun()
    else:
        st.success("✅ Simulation complete.")

with col_reset:
    if st.button("🔄 Reset Simulation", use_container_width=True):
        for key in ['initialized', 'arf_model', 'step', 'history', 'pretrain_perf']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

st.divider()


if st.session_state.step == 0:
    st.write("No batches revealed yet.")
else:
    latest = st.session_state.history[-1]
    batch_idx = latest['batch_idx']
    batch_num = batch_idx + 1
    prob_pred = latest['probs']

    X_batch = batches_X[batch_idx].reset_index(drop=True)
    y_batch = batches_y[batch_idx].reset_index(drop=True)

    is_drift = batch_idx >= DRIFT_START_BATCH
    if is_drift:
        if batch_idx in DRIFT_STRENGTHS:
            strength = DRIFT_STRENGTHS[batch_idx]
        else:
            strength = max(DRIFT_STRENGTHS.values()) if DRIFT_STRENGTHS else 0.0

        st.subheader(f"Month {batch_num}: Drifted (strength {strength:.0%})")
        st.caption(
            f"{strength:.0%} of the month-to-month contract and fibre-optic internet churners in this batch "
            "have been replaced with customers on long-contract customers paying by electronic "
            "check, whose outcome has been switched to true (churn). The total number of "
            "churners is unchanged, so only the type of churners has changed."
        )
    else:
        st.subheader(f"Month {batch_num}: Stable")

    results_df = pd.DataFrame({
        "Customer ID": X_batch.index,
        "Churn Probability": prob_pred,
        "Risk Level": compute_risk_levels(prob_pred),
        "Actual Churn": y_batch.map({0: "No", 1: "Yes"})
    })

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(results_df))
    col2.metric("High Risk", (results_df["Risk Level"] == "High").sum())
    col3.metric("Medium Risk", (results_df["Risk Level"] == "Medium").sum())
    col4.metric("Low Risk", (results_df["Risk Level"] == "Low").sum())

    st.divider()

    left_col, right_col = st.columns([2, 1])

    with left_col:
        def color_risk(val):
            colors = {"High": "background-color: #ffcccc",
                      "Medium": "background-color: #fff3cd",
                      "Low": "background-color: #d4edda"}
            return colors.get(val, "")

        st.markdown("**Customer Risk Table**")

        st.caption("Every customer is ranked by predicted churn probability. "
                   "The model produced these predictions before seeing any of their actual "
                   "outcomes. The final column shows their true label.")

        styled = results_df.sort_values("Churn Probability", ascending=False).style.map(
            color_risk, subset=["Risk Level"]
        ).format({"Churn Probability": "{:.1%}"})
        st.dataframe(styled, use_container_width=True, height=400)

        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download this batch as CSV",
            data=csv,
            file_name=f"churn_predictions_batch_{batch_num}.csv",
            mime="text/csv",
        )

    st.divider()

    st.markdown("**SHAP explainability (via LightGBM )**")

    st.caption("SHAP shows how each feature pushed this customer's prediction up or "
               "down from the average (0.314). Red bars increase predicted risk, blue bars "
               "reduce it, and together they account for the prediction exactly.\n\n "
               "Explanations come from LightGBM model trained on "
               "the same features, as the streaming model does not support this method.")

    col_select, _ = st.columns([1, 2])
    with col_select:
        customer_choice = st.selectbox(
            "Select a customer",
            options=results_df.sort_values("Churn Probability", ascending=False)["Customer ID"].tolist(),
            format_func=lambda x: f"Customer {x} - risk: {results_df.loc[results_df['Customer ID']==x, 'Churn Probability'].values[0]:.1%}"
        )

    shap_values_customer = explainer.shap_values(X_batch.iloc[[customer_choice]])
    if isinstance(shap_values_customer, list):
        shap_values_customer = shap_values_customer[1]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(np.atleast_1d(expected_value)) > 1 else expected_value[0]

    explanation = shap.Explanation(
        values=shap_values_customer[0],
        base_values=expected_value,
        data=X_batch.iloc[customer_choice].values,
        feature_names=X_batch.columns.tolist()
    )

    fig, ax = plt.subplots(figsize=(4, 2.5))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.tight_layout()
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.markdown("**ARF Performance Trajectory**")

    st.caption("Model performance from its first ever batch through to the present. "
               "As the models moves through the new coming batches, values of each metric is calculated and shown in the plot.\n\n"
               "The black line marks the end of initial training; the red dashed line "
               "marks where the drift begins. Early batches are weak because the model "
               "had seen very little data at that point.")

    pretrain_df = pd.DataFrame(st.session_state.pretrain_perf)

    live_rows = []
    for h in st.session_state.history:
        idx = h['batch_idx']
        p = h['probs']
        y_true = batches_y[idx].reset_index(drop=True)
        y_pred = (p >= ARF_THRESHOLD).astype(int)
        live_rows.append({
            'Batch': N_PRETRAIN_BATCHES + idx + 1,
            'AUC': roc_auc_score(y_true, p) if len(set(y_true)) > 1 else np.nan,
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'F2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'Bal_Acc': balanced_accuracy_score(y_true, y_pred),
        })
    live_df = pd.DataFrame(live_rows)

    perf_df = pd.concat([pretrain_df, live_df], ignore_index=True)

    metrics_to_plot = ['AUC', 'F1', 'F2', 'Precision', 'Recall', 'Bal_Acc']
    titles = ['AUC-ROC', 'F1-score', 'F2-score', 'Precision', 'Recall', 'Balanced Accuracy']

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 7))
    axes2 = axes2.flatten()

    drift_batch_num = N_PRETRAIN_BATCHES + DRIFT_START_BATCH + 1

    for ax, metric, title in zip(axes2, metrics_to_plot, titles):
        ax.plot(perf_df["Batch"], perf_df[metric], '-', color="#1E3A5F", linewidth=1.2)
        ax.axvline(x=N_PRETRAIN_BATCHES + 0.5, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axvline(x=drift_batch_num - 0.5, color='#C0392B', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlim(0, N_PRETRAIN_BATCHES + len(batches_X) + 1)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    fig2.suptitle("Pretraining →  |  Live simulation  (red dashed: drift start)", fontsize=9, color='grey')
    plt.tight_layout()

    col_c, col_d = st.columns([8, 1])
    with col_c:
        st.pyplot(fig2)
    plt.close(fig2)

    perf_csv = perf_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download performance data as CSV",
        data=perf_csv,
        file_name="arf_performance_trajectory.csv",
        mime="text/csv",
    )

    st.caption(f"Batches 1–{N_PRETRAIN_BATCHES}: initial pretraining batches on historical data.\n\n"
               f"Batches {N_PRETRAIN_BATCHES+1}–{N_PRETRAIN_BATCHES+len(batches_X)}: live simulation, "
               "revealed one at a time clicked. Red dashed line marks the start of the drifted period.")
