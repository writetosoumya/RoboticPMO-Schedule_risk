
Robotic PMO – Schedule Risk (10‑day) Prototype

Files
- tasks_static_10d.csv
- baseline_risk_scoring.py
- train_model_10d.py
- app_streamlit.py

Quickstart
1) Baseline (no ML)
   python baseline_risk_scoring.py tasks_static_10d.csv

2) Train & evaluate ML model
   pip install scikit-learn pandas numpy
   python train_model_10d.py

3) Optional UI
   pip install streamlit scikit-learn pandas numpy
   streamlit run app_streamlit.py
