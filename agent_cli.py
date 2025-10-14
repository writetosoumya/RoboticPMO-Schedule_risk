
import sys, re, pandas as pd, numpy as np

def explain_row(r):
    reasons = []
    if r.get('status') in ['Blocked','At Risk']:
        reasons.append(f"status {r['status']}")
    if r.get('blocker_count',0) > 0:
        reasons.append(f"{int(r['blocker_count'])} blocker(s)")
    if r.get('risk_count',0) > 0:
        reasons.append(f"{int(r['risk_count'])} risk(s)")
    if r.get('estimate_drift_days',0) > 0:
        reasons.append(f"estimate drift {int(r['estimate_drift_days'])}d")
    if r.get('owner_on_time_ratio',1) < 0.75:
        reasons.append(f"owner on-time {r['owner_on_time_ratio']:.2f}")
    return "; ".join(reasons) or "no major risk signals"

def parse_window(text):
    m = re.search(r'(?:next|within)\s+(\d+)\s+day', text, re.I)
    return int(m.group(1)) if m else 10

def parse_bucket(text):
    if re.search(r'\bhigh\b', text, re.I): return 'High'
    if re.search(r'\bmedium\b', text, re.I): return 'Medium'
    if re.search(r'\blow\b', text, re.I): return 'Low'
    return None

def parse_task_id(text):
    m = re.search(r'\bT(\d{4})\b', text, re.I)
    return f"T{m.group(1)}" if m else None

def main():
    df = pd.read_csv('tasks_static_10d.csv')
    due = df[df['due_within_10_days']==1].copy()
    if due.empty:
        print("No items due within 10 days.")
        return
    if 'risk_probability' not in due.columns:
        due['risk_probability'] = (
            due['blocker_count']*0.08 +
            due['risk_count']*0.06 +
            np.maximum(0, due['estimate_drift_days'])*0.05 +
            (1 - due['owner_on_time_ratio'])*0.5 +
            (due['time_in_state_days']/60)*0.2 +
            due['days_since_last_update']/20*0.15 +
            (due['dependencies_in']/8)*0.1 +
            due['priority'].map({'Low':0,'Medium':0.02,'High':0.05,'Critical':0.07}).fillna(0) +
            due['status'].map({'Blocked':0.2,'At Risk':0.2}).fillna(0) +
            0.15
        ).clip(0,1)
    if 'risk_bucket' not in due.columns:
        due['risk_bucket'] = pd.cut(due['risk_probability'], bins=[-0.01,0.33,0.66,1.0], labels=['Low','Medium','High'])
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        print('Usage: python agent_cli.py "Which tasks are high risk in the next 7 days?"')
        return

    window = parse_window(q)
    dfw = due[due['days_to_due'].between(0, window)].copy()
    task = parse_task_id(q)
    if task:
        row = dfw[dfw['task_id'].str.upper() == task.upper()].head(1)
        if row.empty:
            print(f"Task {task} not found within {window} days.")
            return
        r = row.iloc[0].to_dict()
        why = explain_row(r)
        print(f"{task} risk={r['risk_probability']:.2f} ({r['risk_bucket']}) | {r['project']} | owner {r['owner']} | due in {r['days_to_due']}d")
        print("Because:", why)
        return

    bucket = parse_bucket(q)
    if bucket:
        dfw = dfw[dfw['risk_bucket']==bucket]
    if re.search(r'\bhow many\b|\bcount\b', q, re.I):
        print(len(dfw))
        return
    print(dfw.sort_values('risk_probability', ascending=False).head(10)[['task_id','project','owner','days_to_due','risk_probability','risk_bucket']].to_string(index=False))

if __name__ == "__main__":
    main()
