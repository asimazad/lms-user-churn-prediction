# LMS User Churn Prediction

A complete educational machine-learning demonstration project.

**Data warning:** The included CSV is synthetic, not real LMS data. Replace it with properly authorized historical LMS data before making real-world claims.

## Contents
- `data/`: synthetic LMS-style engagement dataset
- `src/train_model.py`: training and evaluation script
- `outputs/`: model metrics, user-level predictions, feature importance and figures
- `REPORT.md`: project report

## Run
```bash
pip install -r requirements.txt
python src/train_model.py
```

Target: `churn` (1 = churned, 0 = retained).
