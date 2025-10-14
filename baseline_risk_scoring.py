
import sys, pandas as pd, numpy as np

def score_row(r):
    w = 0
    w += r['blocker_count'] * 8
    w += r['risk_count'] * 6
    w += max(0, r['estimate_drift_days']) * 5
    w += (1 - r['owner_on_time_ratio']) * 50
    w += (r['time_in_state_days'] / 60) * 20
    w += 20 if r['status'] in ['Blocked','At Risk'] else 0
    w += 8 if r['priority'] in ['High','Critical'] else 0
    w += (r['days_since_last_update']/20)*15
    w += (r['dependencies_in']/8)*10
    return w

def normalize(scores):
    arr = np.array(scores)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

def main(csv_path):
    df = pd.read_csv(csv_path)
    subset = df[df['due_within_10_days'] == 1].copy()
    if subset.empty:
        print('No items due within 10 days in dataset.')
        return
    subset['risk_score_raw'] = subset.apply(score_row, axis=1)
    subset['risk_score_0_1'] = normalize(subset['risk_score_raw'])
    subset['risk_bucket'] = pd.cut(subset['risk_score_0_1'], bins=[-0.01,0.33,0.66,1.0], labels=['Low','Medium','High'])
    cols = ['task_id','project','owner','days_to_due','percent_complete','status','priority',
            'blocker_count','risk_count','estimate_drift_days','owner_on_time_ratio',
            'time_in_state_days','days_since_last_update','dependencies_in',
            'risk_score_0_1','risk_bucket','label_will_slip_10d']
    print(subset[cols].sort_values('risk_score_0_1', ascending=False).head(25).to_string(index=False))

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv)>1 else 'tasks_static_10d.csv'
    main(path)
