"""Train all dashboard models and save them as .pkl files (optional step).

Run from inside the dashboard folder:   python train_models.py
The app also trains the models by itself if no .pkl files are found.
"""
import joblib
import forecast_lib as fl

raw = fl.load_raw()
df = fl.build_features(raw)
hist, train, test, future = fl.split_data(df)

fl.MODEL_DIR.mkdir(exist_ok=True)
print("Training the evaluation models (data before 2024)...")
joblib.dump(fl.train_models(train), fl.MODEL_DIR / "eval_models.pkl")
print("Training the forecast models (all data)...")
joblib.dump(fl.train_models(hist), fl.MODEL_DIR / "full_models.pkl")
print("Saved to", fl.MODEL_DIR)
