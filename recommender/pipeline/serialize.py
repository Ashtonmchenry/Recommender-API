from pathlib import Path
import joblib

def save_model(model, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path

def load_model(path: str):
    return joblib.load(path)
