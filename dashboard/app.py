"""Sri Lanka Tourism Forecaster: interactive dashboard for the ensemble project.

Run locally:  streamlit run dashboard/app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

import forecast_lib as fl

st.set_page_config(page_title="Sri Lanka Tourism Forecaster", page_icon="🌴", layout="wide")


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


@st.cache_resource(show_spinner=False)
def get_everything():
    raw = fl.load_raw()
    df = fl.build_features(raw)
    hist, train, test, future = fl.split_data(df)
    eval_models, how_eval = fl.load_or_train("eval", train)     # trained before 2024
    full_models, how_full = fl.load_or_train("full", hist)      # trained on all history
    return {"raw": raw, "hist": hist, "train": train, "test": test, "future": future,
            "eval": eval_models, "full": full_models, "how": (how_eval, how_full)}


with st.spinner("Loading the models (the first start can take about a minute)..."):
    S = get_everything()

train, test, future = S["train"], S["test"], S["future"]
EVAL, FULL = S["eval"], S["full"]
names = list(EVAL.keys())
fams = {n: f for n, (f, _) in fl.make_models().items()}
boost_name = [n for n in names if fams[n] == "Boosting"][0]
best_hybrid = "Hybrid: Linear + Voting (RF+ET+GB)"
month_labels = [d.strftime("%b %Y") for d in test["date"]]

# ------------------------------------------------------------------ header
st.title("🌴 Sri Lanka Tourism Forecaster")
st.markdown(
    "**Can machine learning forecast how many tourists arrive next month, and can we understand *why*?** "
    "This dashboard lets you test bagging, boosting, neural-network, stacking and hybrid models on "
    "official Sri Lankan data (SLTDA arrivals and CBSL exchange rate, 2010 to 2026)."
)

with st.sidebar:
    st.header("Meet the models")
    st.markdown(
        "- 🌳 **Bagging (Random Forest):** a *crowd* of trees that vote independently. Stable, but it only "
        "knows values it has seen before.\n"
        f"- 🔁 **Boosting ({boost_name}):** an *apprentice chain*. Each tree fixes the mistakes of the one "
        "before it.\n"
        "- 🧠 **ANN:** a small neural network that learns smooth curves.\n"
        "- 🧩 **Blend and Stacking:** a *committee* of different models, combined by averaging or by a "
        "manager model (Ridge).\n"
        "- 🧭 **Hybrid:** a *compass plus a local guide*. A linear model gives the direction (the trend) and "
        "the ensemble adds the local knowledge (the monsoon seasons)."
    )
    st.divider()
    st.caption("Models: " + S["how"][0] + " (evaluation set); " + S["how"][1] + " (forecast set).")
    st.caption("Evaluation models are trained on Jan 2011 to Dec 2023 only. The forecast models use all data "
               "up to " + S["hist"]["date"].max().strftime("%b %Y") + ".")

tab1, tab2, tab3, tab4 = st.tabs(["1. Time machine", "2. Glass ceiling and what-if lab",
                                  "3. Next month forecast", "4. Scoreboard and why"])

# ------------------------------------------------------------------ tab 1
with tab1:
    st.subheader("Time machine: go back to a month and see who predicted it best")
    st.caption("Models in this tab were trained only on data before 2024, so every month below is a real, "
               "unseen future for them.")
    c1, c2 = st.columns([1, 2])
    with c1:
        label = st.selectbox("Choose a month", month_labels, index=min(25, len(month_labels) - 1))
        shown = st.multiselect("Models to draw on the chart", names,
                               default=["Random Forest", boost_name, best_hybrid])
    idx = month_labels.index(label)
    row = test.iloc[idx]
    X1 = pd.DataFrame([row[fl.FEATURES]])
    p1 = {n: float(v[0]) for n, v in fl.predict_all(EVAL, X1).items()}
    actual = float(row["arrivals"])
    tbl = pd.DataFrame({"Predicted": p1}).assign(Error=lambda d: d["Predicted"] - actual)
    tbl["Error %"] = tbl["Error"] / actual * 100
    best_n = tbl["Error"].abs().idxmin()
    ml_vals = [v for n, v in p1.items() if not n.startswith("Naive")]
    with c1:
        st.metric("Actual arrivals", f"{actual:,.0f}")
        st.metric("Closest model", best_n, f"error {tbl.loc[best_n, 'Error']:+,.0f}")
        spread = (max(ml_vals) - min(ml_vals)) / np.median(ml_vals) * 100
        st.metric("Model disagreement", f"{spread:.0f}%", help="Gap between the highest and lowest model, "
                  "as a share of the median prediction. A big gap means this month is hard to call.")
    with c2:
        preds_all = fl.predict_all(EVAL, test)
        show(fl.plot_timeline(test, preds_all, shown, highlight=row["date"]))
    st.dataframe(tbl.round(0).astype({"Predicted": int, "Error": int}).assign(**{"Error %": tbl["Error %"].round(1)}))

    hy = EVAL[best_hybrid]
    trend, corr = [float(x[0]) for x in hy.parts(X1)]
    st.markdown(f"**Why did the best hybrid say {trend + corr:,.0f}?**  The linear *compass* (trend) says "
                f"**{trend:,.0f}**, and the trees add a local correction of **{corr:+,.0f}** "
                "(mostly the seasonal pattern). Total = trend + correction.")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.subheader("Glass ceiling: why plain tree models fail at record months")
    st.markdown("A Random Forest predicts by **averaging values it saw in training**, so it cannot go above "
                "them. Push last month's arrivals up and watch which models follow.")
    c1, c2 = st.columns([1, 2])
    with c1:
        label2 = st.selectbox("Month to predict", month_labels, index=min(25, len(month_labels) - 1), key="m2")
        row2 = test.iloc[month_labels.index(label2)]
        preset = st.radio("Scenario", ["Use the slider", "Sudden crisis: -50%", "Sudden drop: -30%",
                                       "Record boom: +40%"])
        base = float(row2["lag_1"])
        slider_val = st.slider("Last month's arrivals", 20000, 450000, int(base), step=5000)
        mult = {"Sudden crisis: -50%": 0.5, "Sudden drop: -30%": 0.7, "Record boom: +40%": 1.4}.get(preset)
        lag1 = base * mult if mult else float(slider_val)
        fx_pct = st.slider("Rupee change vs last month's rate (%)", -30, 30, 0) / 100
        st.caption(f"Real last month: {base:,.0f}.  What-if value used: {lag1:,.0f}.")
    sweep_names = ["Random Forest", boost_name, "ANN (neural network)", "Hybrid: Linear + Random Forest", best_hybrid]
    vals = np.linspace(20000, 450000, 40)
    with c2:
        sw = fl.sweep_lag1(EVAL, row2, vals, sweep_names)
        show(fl.plot_sweep(sw, vals, float(train["arrivals"].max()), current=lag1))
    Xw = pd.DataFrame([fl.apply_whatif(row2, lag1=lag1, fx_pct=fx_pct)])
    pw = fl.predict_all(EVAL, Xw)
    st.markdown("**Predictions for your what-if scenario**")
    out = pd.DataFrame({"Predicted arrivals": {n: int(v[0]) for n, v in pw.items()}})
    st.dataframe(out)
    st.info("What to notice: the Random Forest flattens out near the top of what it has seen, the hybrid keeps "
            "rising because its linear part extrapolates, and the exchange-rate slider barely changes anything "
            "(the exchange rate is a weak predictor in this data).")

# ------------------------------------------------------------------ tab 3
with tab3:
    next_date = future["date"].iloc[0]
    st.subheader(f"Next month forecast: {next_date.strftime('%B %Y')}")
    st.caption("These models are trained on all months up to the latest data. This is a forecast, not a fact.")
    Xf = future[fl.FEATURES]
    pf = {n: float(v[0]) for n, v in fl.predict_all(FULL, Xf).items()}
    ml = {n: v for n, v in pf.items() if not n.startswith("Naive")}
    med, lo, hi = np.median(list(ml.values())), min(ml.values()), max(ml.values())
    spread_f = (hi - lo) / med * 100
    level = "High" if spread_f < 10 else ("Medium" if spread_f < 20 else "Low")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best hybrid says", f"{pf[best_hybrid]:,.0f}")
    m2.metric("Median of all models", f"{med:,.0f}")
    m3.metric("Range (low to high)", f"{lo / 1000:,.0f}k to {hi / 1000:,.0f}k")
    m4.metric("Model agreement", level, f"gap {spread_f:.0f}%", delta_color="off")
    show(fl.plot_forecast_bars(pf))
    trend_f, corr_f = [float(x[0]) for x in FULL[best_hybrid].parts(Xf)]
    st.markdown(f"**Inside the best hybrid:** trend (compass) **{trend_f:,.0f}**, seasonal correction "
                f"(local guide) **{corr_f:+,.0f}**. The inputs are last month's arrivals "
                f"({float(future['lag_1'].iloc[0]):,.0f}), the month before ({float(future['lag_2'].iloc[0]):,.0f}), "
                f"the same month last year ({float(future['lag_12'].iloc[0]):,.0f}) and the monsoon season "
                f"({fl.SEASON_LABELS[fl.monsoon_season(next_date.month)]}).")
    st.warning("Model agreement is the gap between models, not a statistical confidence interval. "
               "Sudden events, such as a crisis or a new flight route, cannot be predicted from past data.")

# ------------------------------------------------------------------ tab 4
with tab4:
    st.subheader("Scoreboard: Jan 2024 to Jul 2026 (31 months the models never saw)")
    sc = fl.score_models(EVAL, test)
    show(fl.plot_scores(sc))
    show_sc = sc.copy()
    show_sc[["MAE", "RMSE"]] = show_sc[["MAE", "RMSE"]].round(0).astype(int)
    show_sc["MAPE %"] = show_sc["MAPE %"].round(1)
    st.dataframe(show_sc)
    fam_best = sc.groupby("Family")["MAE"].min().sort_values()
    st.markdown("**Best model in each family:** " + ", ".join(
        f"{f} ({sc[(sc['Family'] == f)]['MAE'].idxmin()}, {v:,.0f})" for f, v in fam_best.items()))
    st.markdown("**Why do the hybrids win?** Tree models can only average what they have seen, so they miss "
                "record-high months. The linear trend fixes that, and the trees then learn the monsoon-season "
                "pattern that the trend alone cannot. Neural networks extrapolate better than trees, which is "
                "why the ANN beats the plain Random Forest here.")

    st.subheader("Error by monsoon season")
    st.caption("Average error (tourists) for each season in the test period.")
    sr = fl.season_report(EVAL, test).astype(int)
    st.dataframe(sr)

    st.subheader("What matters most?")
    rf_imp = pd.Series(EVAL["Random Forest"].feature_importances_, index=fl.FEATURES).sort_values()
    st.bar_chart(rf_imp)
    st.caption("Random Forest feature importance: last month's arrivals dominate.")
    from pathlib import Path
    fig_dir = fl.ROOT / "figures"
    cols = st.columns(2)
    for col, (fname, cap) in zip(cols, [("09_group_importance.png", "Group importance of the best hybrid"),
                                        ("10_shap_summary.png", "SHAP: what the trees add to the trend")]):
        p = fig_dir / fname
        if p.exists():
            col.image(str(p), caption=cap)
    p = fig_dir / "08_backtest_mean_error.png"
    if p.exists():
        st.image(str(p), caption="Five-window backtest: hybrids are best on average, not in every window")

st.caption("Data: Sri Lanka Tourism Development Authority and Central Bank of Sri Lanka. "
           "Built for a university ensemble-learning project.")
