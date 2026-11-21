"""Forecasting engine for the Sri Lanka tourism dashboard.

Everything that is not user-interface code lives here, so it can be tested on its own.
"""
from pathlib import Path
from collections import OrderedDict
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              VotingRegressor, StackingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_PATH = ROOT / "data" / "tourism_monthly_2010_2026.csv"
MODEL_DIR = HERE / "models"

FEATURES = ["month", "quarter", "month_sin", "month_cos",
            "season_SW_monsoon", "season_NE_monsoon", "season_Inter_monsoon", "shock",
            "lag_1", "lag_2", "lag_3", "lag_12", "roll3", "roll12", "fx_lag1", "fx_chg_lag1"]
SEASON_COLORS = {"SW_monsoon": "tab:blue", "NE_monsoon": "tab:orange", "Inter_monsoon": "tab:green"}
SEASON_LABELS = {"SW_monsoon": "South-West monsoon (May-Sep)",
                 "NE_monsoon": "North-East monsoon (Dec-Feb)",
                 "Inter_monsoon": "Inter-monsoon (Mar-Apr, Oct-Nov)"}
FAMILY_COLORS = {"Baseline": "gray", "Bagging": "tab:green", "Boosting": "tab:orange",
                 "Neural network": "tab:red", "Blend": "tab:olive", "Stacking": "tab:purple",
                 "Hybrid": "tab:blue"}


# ---------------------------------------------------------------- data and features
def monsoon_season(m):
    if m in (5, 6, 7, 8, 9):
        return "SW_monsoon"
    if m in (12, 1, 2):
        return "NE_monsoon"
    return "Inter_monsoon"


def load_raw(path=DATA_PATH):
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def build_features(raw):
    """Same features as the notebook. One empty row is added for the next, unknown month."""
    df = raw.copy()
    nxt = df["date"].max() + pd.offsets.MonthBegin(1)
    df = pd.concat([df, pd.DataFrame({"date": [nxt], "arrivals": [np.nan], "usd_lkr": [np.nan]})],
                   ignore_index=True)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["season"] = df["month"].map(monsoon_season)
    for s in ("SW_monsoon", "NE_monsoon", "Inter_monsoon"):
        df[f"season_{s}"] = (df["season"] == s).astype(int)
    df["shock"] = ((df["date"] >= "2019-04-01") & (df["date"] <= "2022-12-01")).astype(int)
    for k in (1, 2, 3, 12):
        df[f"lag_{k}"] = df["arrivals"].shift(k)
    df["roll3"] = df["arrivals"].shift(1).rolling(3).mean()
    df["roll12"] = df["arrivals"].shift(1).rolling(12).mean()
    df["fx_lag1"] = df["usd_lkr"].shift(1)
    last_known = df["usd_lkr"].last_valid_index()
    chg = df["usd_lkr"].iloc[: last_known + 1].pct_change()
    df["fx_chg_lag1"] = chg.reindex(df.index).shift(1)
    return df


def split_data(df):
    hist = df[df["arrivals"].notna()].dropna(subset=FEATURES).reset_index(drop=True)
    future = df[df["arrivals"].isna()].reset_index(drop=True)
    train = hist[hist["date"] < "2024-01-01"].reset_index(drop=True)
    test = hist[hist["date"] >= "2024-01-01"].reset_index(drop=True)
    return hist, train, test, future


# ---------------------------------------------------------------- models
class HybridRegressor(BaseEstimator, RegressorMixin):
    """Linear regression learns the trend, an ensemble learns what the trend missed."""
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y):
        self.trend_ = LinearRegression().fit(X, y)
        self.model_ = clone(self.model).fit(X, y - self.trend_.predict(X))
        return self

    def predict(self, X):
        return self.trend_.predict(X) + self.model_.predict(X)

    def parts(self, X):
        return self.trend_.predict(X), self.model_.predict(X)


def _ann():
    net = make_pipeline(StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=3000, random_state=42))
    return TransformedTargetRegressor(regressor=net, transformer=StandardScaler())


def _booster():
    try:
        from catboost import CatBoostRegressor
        return "CatBoost", CatBoostRegressor(iterations=300, learning_rate=0.05, depth=4,
                                             random_seed=42, verbose=0)
    except Exception:
        return "Gradient Boosting", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                                               subsample=0.8, random_state=42)


def make_models():
    """name -> (family, unfitted model)"""
    bname, boost = _booster()
    rf = lambda: RandomForestRegressor(n_estimators=300, random_state=42)
    et = lambda: ExtraTreesRegressor(n_estimators=200, random_state=42)
    gb = lambda: GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                                           subsample=0.8, random_state=42)
    voting = lambda: VotingRegressor([("rf", rf()), ("et", et()), ("gb", gb())])
    m = OrderedDict()
    m["Linear regression"] = ("Baseline", LinearRegression())
    m["Random Forest"] = ("Bagging", rf())
    m[bname] = ("Boosting", clone(boost))
    m["ANN (neural network)"] = ("Neural network", _ann())
    m[f"Blend: RF + {bname}"] = ("Blend", VotingRegressor([("rf", rf()), ("b", clone(boost))]))
    m[f"Blend: {bname} + ANN"] = ("Blend", VotingRegressor([("b", clone(boost)), ("ann", _ann())]))
    m[f"Stacking: RF + {bname} + ANN"] = ("Stacking", StackingRegressor(
        estimators=[("rf", rf()), ("b", clone(boost)), ("ann", _ann())], final_estimator=RidgeCV(), cv=5))
    m["Hybrid: Linear + Random Forest"] = ("Hybrid", HybridRegressor(rf()))
    m[f"Hybrid: Linear + {bname}"] = ("Hybrid", HybridRegressor(clone(boost)))
    m["Hybrid: Linear + Voting (RF+ET+GB)"] = ("Hybrid", HybridRegressor(voting()))
    return m


def family_of(name):
    return {n: f for n, (f, _) in make_models().items()}.get(name, "Baseline")


def train_models(train_df):
    fitted = OrderedDict()
    for name, (fam, est) in make_models().items():
        fitted[name] = clone(est).fit(train_df[FEATURES], train_df["arrivals"])
    return fitted


def load_or_train(kind, train_df):
    """kind = 'eval' (trained before 2024) or 'full' (trained on all history).
    Loads models/<kind>_models.pkl if it exists and works, otherwise trains and saves it."""
    path = MODEL_DIR / f"{kind}_models.pkl"
    if path.exists():
        try:
            return joblib.load(path), "loaded from saved .pkl file"
        except Exception:
            pass
    fitted = train_models(train_df)
    try:
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(fitted, path)
    except Exception:
        pass
    return fitted, "trained now (and saved as .pkl)"


# ---------------------------------------------------------------- predictions
def predict_all(models, X):
    out = OrderedDict()
    out["Naive (last month)"] = np.asarray(X["lag_1"], dtype=float)
    for name, m in models.items():
        out[name] = np.asarray(m.predict(X[FEATURES]), dtype=float)
    return out


def score_models(models, test_df):
    preds = predict_all(models, test_df)
    preds["Seasonal naive (same month last year)"] = np.asarray(test_df["lag_12"], dtype=float)
    y = test_df["arrivals"].values
    rows = {}
    for n, p in preds.items():
        rows[n] = {"MAE": mean_absolute_error(y, p),
                   "RMSE": mean_squared_error(y, p) ** 0.5,
                   "MAPE %": float(np.mean(np.abs((y - p) / y)) * 100)}
    fam = {n: f for n, (f, _) in make_models().items()}
    t = pd.DataFrame(rows).T
    t["Family"] = [fam.get(n, "Baseline") for n in t.index]
    return t.sort_values("MAE")


def season_report(models, test_df):
    preds = predict_all(models, test_df)
    y = test_df["arrivals"].values
    rep = {}
    for n, p in preds.items():
        err = pd.Series(np.abs(y - p))
        rep[n] = err.groupby(test_df["season"].values).mean()
    return pd.DataFrame(rep).T.round(0)


def apply_whatif(row, lag1=None, fx_pct=0.0):
    """Change last month's arrivals and/or the exchange rate, keeping the related features consistent."""
    r = row[FEATURES].astype(float).copy()
    if lag1 is not None and lag1 != r["lag_1"]:
        old = r["lag_1"]
        r["lag_1"] = lag1
        r["roll3"] = (lag1 + r["lag_2"] + r["lag_3"]) / 3
        r["roll12"] = r["roll12"] + (lag1 - old) / 12
    if fx_pct:
        r["fx_chg_lag1"] = (1 + r["fx_chg_lag1"]) * (1 + fx_pct) - 1
        r["fx_lag1"] = r["fx_lag1"] * (1 + fx_pct)
    return r


def sweep_lag1(models, row, values, names):
    out = {n: [] for n in names}
    for v in values:
        X = pd.DataFrame([apply_whatif(row, lag1=v)])
        for n in names:
            out[n].append(float(models[n].predict(X[FEATURES])[0]))
    return out


# ---------------------------------------------------------------- charts
def shade_seasons(ax, dates):
    for d, s in zip(dates, [monsoon_season(x.month) for x in dates]):
        ax.axvspan(d, d + pd.offsets.MonthBegin(1), color=SEASON_COLORS[s], alpha=0.07, linewidth=0)


def season_legend():
    return [Patch(color=SEASON_COLORS[k], alpha=0.25, label=SEASON_LABELS[k]) for k in SEASON_COLORS]


def plot_timeline(test_df, preds, show, highlight=None):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    shade_seasons(ax, list(test_df["date"]))
    ax.plot(test_df["date"], test_df["arrivals"], color="black", linewidth=2.6, label="Actual")
    fams = {n: f for n, (f, _) in make_models().items()}
    for n in show:
        ax.plot(test_df["date"], preds[n], linewidth=1.6, alpha=0.9, label=n,
                color=FAMILY_COLORS.get(fams.get(n, "Baseline")))
    if highlight is not None:
        ax.axvline(highlight, color="red", linestyle="--", alpha=0.6)
    ax.set_ylabel("Tourist arrivals per month")
    ax.set_title("Actual vs predicted arrivals (background = monsoon season)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + season_legend(), fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sweep(sweep, values, train_max, current=None):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fams = {n: f for n, (f, _) in make_models().items()}
    for n, ys in sweep.items():
        ax.plot(values, ys, linewidth=2, label=n, color=FAMILY_COLORS.get(fams.get(n, "Baseline")))
    ax.axvline(train_max, color="gray", linestyle=":", label=f"Highest month in training ({train_max:,.0f})")
    if current is not None:
        ax.axvline(current, color="red", linestyle="--", alpha=0.6, label="Your chosen value")
    ax.plot(values, values, color="black", linewidth=0.8, alpha=0.4, label="'Same as last month'")
    ax.set_xlabel("Last month's arrivals (what-if value)")
    ax.set_ylabel("Predicted arrivals for the month shown")
    ax.set_title("The glass ceiling: what each model predicts as last month's arrivals rise")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_scores(table):
    t = table.sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(t) + 1.4))
    ax.barh(t.index, t["MAE"], color=[FAMILY_COLORS.get(f, "gray") for f in t["Family"]])
    ax.set_xlabel("MAE: average error in tourists per month (smaller is better)")
    ax.set_title("Scoreboard on the test period (Jan 2024 to Jul 2026)")
    ax.legend(handles=[Patch(color=c, label=f) for f, c in FAMILY_COLORS.items() if f in set(t["Family"])],
              loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def plot_forecast_bars(preds_next):
    names = list(preds_next.keys())
    vals = [preds_next[n] for n in names]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(names) + 1.4))
    fams = {n: f for n, (f, _) in make_models().items()}
    ax.barh([names[i] for i in order], [vals[i] for i in order],
            color=[FAMILY_COLORS.get(fams.get(names[i], "Baseline")) for i in order])
    ax.set_xlabel("Predicted tourist arrivals")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig
