
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import re

# ---- Voice Input Helpers ----
def transcribe_audio_bytes(wav_bytes, language='en'):
    import tempfile, io
    import numpy as np
    import soundfile as sf
    import whisper
    # Write bytes to a temp wav and transcribe
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        model = whisper.load_model("tiny.en")
        result = model.transcribe(tmp.name, language=language)
        return result.get("text", "").strip()

st.set_page_config(page_title="Robotic PMO – Schedule Risk (10‑day)", layout="wide")
st.title("Robotic PMO – Schedule Risk (10‑day Horizon)")

@st.cache_data
def load_data():
    df = pd.read_csv('tasks_static_10d.csv')
    return df

def train_model_and_score(df):
    due = df[df['due_within_10_days']==1].copy()
    if due.empty:
        return due, None, None

    y = due['label_will_slip_10d']
    feature_cols = [
        'percent_complete','planned_effort_hours','actual_effort_hours',
        'reopen_count','change_count','blocker_count','risk_count',
        'dependencies_in','dependencies_out','owner_on_time_ratio',
        'estimate_drift_days','time_in_state_days','days_since_last_update',
        'days_to_due'
    ]
    X_base = due[feature_cols]
    X_cat = pd.get_dummies(due[['status','priority']], drop_first=True)
    X = pd.concat([X_base, X_cat], axis=1)

    if len(due) > 5 and y.nunique() > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        gb = GradientBoostingClassifier(random_state=42)
        clf = CalibratedClassifierCV(gb, method='isotonic', cv=3)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X)[:,1]
    else:
        # Fallback: heuristic probability from a simple linear combo
        proba = (
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
        ).clip(0,1).values
        clf = None

    due['risk_probability'] = proba
    due['risk_bucket'] = pd.cut(proba, bins=[-0.01,0.33,0.66,1.0], labels=['Low','Medium','High'])
    return due, clf, X.columns.tolist()

def explain_row(r):
    # Lightweight heuristic explanation for why a task is risky.
    reasons = []
    if r.get('status') in ['Blocked','At Risk']:
        reasons.append(f"status is {r['status']}")
    if r.get('blocker_count',0) > 0:
        reasons.append(f"{int(r['blocker_count'])} blocker(s)")
    if r.get('risk_count',0) > 0:
        reasons.append(f"{int(r['risk_count'])} risk(s) logged")
    if r.get('estimate_drift_days',0) > 0:
        reasons.append(f"estimate drift {int(r['estimate_drift_days'])}d")
    if r.get('owner_on_time_ratio',1) < 0.75:
        reasons.append(f"owner on-time {r['owner_on_time_ratio']:.2f} (low)")
    if r.get('time_in_state_days',0) > 20:
        reasons.append(f"time-in-state {int(r['time_in_state_days'])}d")
    if r.get('days_since_last_update',0) > 7:
        reasons.append(f"stale update {int(r['days_since_last_update'])}d")
    if r.get('dependencies_in',0) >= 4:
        reasons.append(f"high dependencies-in ({int(r['dependencies_in'])})")
    if not reasons:
        reasons.append("no major risk signals detected")
    return "; ".join(reasons)

def parse_window(text):
    m = re.search(r'(?:next|within)\s+(\d+)\s+day', text, re.I)
    return int(m.group(1)) if m else 10

def parse_bucket(text):
    if re.search(r'\bhigh\b', text, re.I): return 'High'
    if re.search(r'\bmedium\b', text, re.I): return 'Medium'
    if re.search(r'\blow\b', text, re.I): return 'Low'
    return None

def parse_breakdown(text):
    if re.search(r'\bby project\b', text, re.I): return 'project'
    if re.search(r'\bby owner\b', text, re.I): return 'owner'
    if re.search(r'\bby status\b', text, re.I): return 'status'
    if re.search(r'\bby priority\b', text, re.I): return 'priority'
    return None

def parse_task_id(text):
    m = re.search(r'\bT(\d{4})\b', text, re.I)
    return f"T{m.group(1)}" if m else None

def handle_query(q, due_df):
    q = q.strip()
    if due_df is None or due_df.empty:
        return "No items due in the specified window.", None

    window = parse_window(q)
    df = due_df.copy()
    df = df[df['days_to_due'].between(0, window)]

    task = parse_task_id(q)
    if task:
        row = df[df['task_id'].str.upper() == task.upper()].head(1)
        if row.empty:
            return f"I couldn't find {task} due within {window} days.", None
        r = row.iloc[0].to_dict()
        why = explain_row(r)
        msg = (f"{task} – risk {r.get('risk_probability',0):.2f} ({r.get('risk_bucket')}) "
               f"| {r.get('project')} | owner {r.get('owner')} | due in {r.get('days_to_due')} day(s).\n"
               f"Because: {why}.")
        return msg, row

    bucket = parse_bucket(q)
    if bucket:
        df = df[df['risk_bucket'] == bucket]

    if re.search(r'\bhow many\b|\bcount\b', q, re.I):
        n = len(df)
        msg = f"{n} task(s) due within {window} days"
        if bucket: msg += f" with {bucket} risk"
        return msg + ".", None

    by = parse_breakdown(q)
    if by:
        grp = df.groupby(by)['task_id'].count().reset_index().rename(columns={'task_id':'count'}).sort_values('count', ascending=False)
        return f"Breakdown by {by} (due within {window} days" + (f", {bucket} risk" if bucket else "") + "):", grp

    topn = 15
    view = df.sort_values('risk_probability', ascending=False).head(topn)
    return (f"Top {len(view)} risky task(s) due within {window} days"
            + (f" (bucket: {bucket})" if bucket else "")
            + ":"), view

df = load_data()
due_df, clf, feat_cols = train_model_and_score(df)
st.caption(f"Dataset rows: {len(df)} | Due in next 10 days: {len(due_df) if due_df is not None else 0}")

tab1, tab2 = st.tabs(["Dashboard", "Ask Savvy (Agentic Q&A)"])

with tab1:
    st.subheader("Top Risks (next 10 days)")
    if due_df is None or due_df.empty:
        st.info("No items due within 10 days.")
    else:
        cols = ['task_id','project','owner','days_to_due','percent_complete','status','priority',
                'blocker_count','risk_count','estimate_drift_days','owner_on_time_ratio','risk_probability','risk_bucket']
        st.dataframe(due_df.sort_values('risk_probability', ascending=False)[cols].head(30), use_container_width=True)
        st.subheader("Download predictions")
        st.download_button("Download CSV", data=due_df[cols].to_csv(index=False), file_name="predictions_ui.csv", mime="text/csv")

with tab2:
    st.subheader("Ask about schedule risks")

    st.markdown("**Use your voice**: Click Record, ask your question, then Stop. We'll transcribe and fill the box.")
    try:
        from st_audiorec import st_audiorec
        wav_audio_data = st_audiorec()
        if wav_audio_data is not None:
            st.info("Transcribing... (tiny.en)")
            try:
                question_text = transcribe_audio_bytes(wav_audio_data)
                if question_text:
                    st.success(f"Transcribed: {question_text}")
                    q = st.text_input("Your question", value=question_text)
                else:
                    st.warning("Could not transcribe audio. Please try again or type your question.")
            except Exception as e:
                st.error(f"Transcription error: {e}")
    except Exception as e:
        st.caption("To enable mic recording: `pip install streamlit-audiorec openai-whisper soundfile`")
        st.caption("After installing, restart Streamlit.")

    st.write("Examples:")
    st.code('''
- Which tasks are high risk in the next 7 days?
- How many medium-risk items are due within 5 days?
- Breakdown by project for high risk within 10 days
- Why T0005?
''', language="markdown")
    q = st.text_input("Your question")
    if st.button("Ask") and q.strip():
        msg, table = handle_query(q, due_df)
        st.write(msg)
        if table is not None and not table.empty:
            st.dataframe(table, use_container_width=True)
