"""Train and evaluate LMS churn models on the included synthetic dataset."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "lms_engagement_synthetic.csv")
X = df.drop(columns=["user_id", "churn"])
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42
    ),
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predicted),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1_score": f1_score(y_test, predicted, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
    })
results_df = pd.DataFrame(results)
out = ROOT / "outputs"
out.mkdir(exist_ok=True)
results_df.to_csv(out / "model_metrics.csv", index=False)
print(results_df.to_string(index=False))
best_name = results_df.sort_values("F1_score", ascending=False).iloc[0]["Model"]
best = models[best_name]
best.fit(X_train, y_train)
all_prob = best.predict_proba(X)[:, 1]
predictions = pd.DataFrame({
    "user_id": df["user_id"],
    "churn_probability": all_prob.round(3),
})
predictions["risk_level"] = pd.cut(
    all_prob, [-0.001, 0.30, 0.60, 1.001],
    labels=["Low", "Medium", "High"]
)
actions = {
    "Low": "Continue normal engagement and offer optional enrichment.",
    "Medium": "Send a supportive reminder and monitor engagement.",
    "High": "Offer a personal check-in and help identify barriers."
}
predictions["recommended_support"] = predictions["risk_level"].astype(str).map(actions)
predictions.to_csv(out / "user_churn_predictions.csv", index=False)
print(f"Selected model by F1-score: {best_name}")
