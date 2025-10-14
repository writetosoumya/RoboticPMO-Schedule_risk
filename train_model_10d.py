
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report, roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

df = pd.read_csv('tasks_static_10d.csv')
df = df[df['due_within_10_days'] == 1].copy()
y = df['label_will_slip_10d']

feature_cols = [
    'percent_complete','planned_effort_hours','actual_effort_hours',
    'reopen_count','change_count','blocker_count','risk_count',
    'dependencies_in','dependencies_out','owner_on_time_ratio',
    'estimate_drift_days','time_in_state_days','days_since_last_update',
    'days_to_due'
]
df_enc = pd.get_dummies(df[['status','priority']], drop_first=True)
X = pd.concat([df[feature_cols], df_enc], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

gb = GradientBoostingClassifier(random_state=42)
clf = CalibratedClassifierCV(gb, method='isotonic', cv=3)
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, proba)
ap = average_precision_score(y_test, proba)
print(f'AUC: {auc:.3f}  |  PR-AUC: {ap:.3f}')

fpr, tpr, thr = roc_curve(y_test, proba)
import numpy as np
j_scores = tpr - fpr
best_t = thr[j_scores.argmax()]
print(f'Chosen threshold: {best_t:.3f}')

pred = (proba >= best_t).astype(int)
print('Confusion matrix:\n', confusion_matrix(y_test, pred))
print('\nClassification report:\n', classification_report(y_test, pred))

out = df.loc[y_test.index, ['task_id','project','owner','days_to_due','percent_complete','status','priority']].copy()
out['risk_probability'] = proba
out['predicted_slip'] = pred
out.sort_values('risk_probability', ascending=False, inplace=True)
out.to_csv('predictions_10d.csv', index=False)
print('\nSaved predictions to predictions_10d.csv')
