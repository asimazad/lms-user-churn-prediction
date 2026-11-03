# LMS User Churn Prediction
## 1. Introduction
This project predicts which users of a Learning Management System (LMS) may stop using the platform. Early identification can help an institution offer timely, personalized academic support.

## 2. Objective
Analyze user login behavior and course engagement, train classification models, compare Random Forest and Gradient Boosting, and produce risk groups with suggested support actions.

## 3. Dataset
A **synthetic demonstration dataset** of 1500 user records was generated because no real LMS data was supplied. It is not collected from real students and must not be presented as real institutional data. The CSV is `data/lms_engagement_synthetic.csv`.

Features:
- `login_days_last_30`: number of days with a login in the last 30 days
- `courses_enrolled`: courses enrolled
- `course_completion_pct`: course completion percentage
- `learning_hours_last_30`: learning hours in the last 30 days
- `days_since_last_login`: recency of last login
- `assignments_submitted_last_30`: assignments submitted in the last 30 days
- `forum_posts_last_30`: forum contributions in the last 30 days
- `churn`: target label (1 = churned, 0 = retained)

For this demonstration, churn is synthetically assigned using a probability influenced by lower activity and engagement. In a real deployment, churn must be defined using a time-based rule (for example, no login for 30 consecutive days) and labels must be built from historical data.

Class counts: retained = 1187, churned = 313.

## 4. Methodology
1. Load and inspect the CSV.
2. Separate predictors from the target label.
3. Split the data into 80% training and 20% testing using stratification and a fixed random seed.
4. Train Random Forest and Gradient Boosting classifiers.
5. Compare models using accuracy, precision, recall, F1-score, and ROC-AUC.
6. Use the selected model to estimate churn probabilities and assign Low (under 0.30), Medium (0.30–0.60), and High (over 0.60) risk bands.

## 5. Model Results
- **Random Forest:** Accuracy 0.773, Precision 0.453, Recall 0.381, F1 0.414, ROC-AUC 0.691
- **Gradient Boosting:** Accuracy 0.793, Precision 0.524, Recall 0.175, F1 0.262, ROC-AUC 0.683

The selected model by F1-score is **Random Forest**. Its test results are: Accuracy **0.773**, Precision **0.453**, Recall **0.381**, F1-score **0.414**, ROC-AUC **0.691**.

These are demonstration results on synthetic data. They do not establish expected performance on real LMS users.

## 6. Visual Analysis
- `outputs/figures/churn_distribution.png`: distribution of the two target classes.
- `outputs/figures/login_days_by_churn.png`: average login days by class.
- `outputs/figures/completion_by_churn.png`: average course completion by class.
- `outputs/figures/confusion_matrix.png`: test-set classification outcomes.
- `outputs/figures/feature_importance.png`: model feature importance.

## 7. Churn Prevention Recommendations
- **High risk:** offer a personal, non-judgmental check-in; ask what is blocking progress; recommend a small achievable next step.
- **Medium risk:** send a supportive reminder, highlight unfinished course activities, and monitor changes in engagement.
- **Low risk:** continue normal communication and provide optional learning resources.

Predictions should guide supportive outreach, not punitive decisions. Staff should review context before contacting a learner.

## 8. Limitations
- The data and target labels are synthetic.
- Synthetic patterns may be easier for a model to learn than real behavior.
- The model does not prove that low engagement causes churn.
- The risk thresholds are illustrative and should be calibrated with real validation data.
- Real LMS deployment requires privacy safeguards, consent/appropriate institutional authority, monitoring for bias, and periodic retraining.

## 9. Conclusion
The project demonstrates an end-to-end churn prediction workflow using LMS-style engagement features. Random Forest and Gradient Boosting were trained and evaluated, and the selected model generated user-level risk estimates and suggested support actions. The next step is to replace the synthetic dataset with approved historical LMS logs and evaluate the model prospectively.

## 10. How to Run
Install dependencies:
```bash
pip install -r requirements.txt
```
Run:
```bash
python src/train_model.py
```
