# Dashboard: Sri Lanka Tourism Forecaster

An interactive Streamlit app that compares bagging, boosting, neural-network, blended, stacked and hybrid models for forecasting monthly tourist arrivals.

**Tabs**
1. **Time machine:** pick any month from Jan 2024 to Jul 2026, see what every model predicted, who was closest, and how much the models disagreed. For the best hybrid it shows the split between the linear trend and the seasonal correction.
2. **Glass ceiling and what-if lab:** change last month's arrivals (or simulate a crisis or a boom) and the exchange rate, and watch how each model reacts. A plain Random Forest cannot predict above the values it has seen; the hybrid can.
3. **Next month forecast:** forecast for the month after the latest data, with a "model agreement" meter.
4. **Scoreboard and why:** error scores, errors by monsoon season, feature importance, SHAP and backtest charts.

**Run locally**
```
pip install -r requirements.txt
streamlit run app.py
```

**Saved models (.pkl)**
The app looks for `models/eval_models.pkl` and `models/full_models.pkl`. If they are missing (or were made with a different library version), it trains the models itself the first time it starts (about one minute) and saves them. To create the files yourself: `python train_models.py`.

**Files**
- `app.py`: the Streamlit interface
- `forecast_lib.py`: features, models, hybrid class, predictions and charts
- `train_models.py`: optional script that saves the .pkl files
