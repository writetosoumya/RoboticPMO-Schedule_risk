#  Robotic PMO – AI-Native Prototype  
### Predictive Schedule Risk Intelligence for Program & Portfolio Management

---

## Overview
**Robotic PMO** is an **AI-native solution** designed to predict schedule risks within a Program & Portfolio Management (PPM) landscape.  
It is purpose-built to operate even in **non-integrated, data-sparse environments**, enabling PMOs and transformation leaders to bring AI-powered foresight into project governance.

This working prototype uses **machine learning and rule-based intelligence** to analyze task-level data, identify at-risk milestones, and generate **predictive insights** for decision-making  without requiring a connected ecosystem.

---

## Key Capabilities
| Category | Description |
|-----------|-------------|
| **Schedule Risk Prediction** | Uses supervised ML and rule-based logic to flag tasks that may slip within 10 days. |
| **Agentic AI Q&A Interface** | Allows users to query schedule risks using natural language (with or without voice). |
| **Explainable Insights** | Offers rationale behind each predicted risk, ensuring transparency for PMO decision-makers. |
| **Modular & Scalable Design** | Built to onboard additional work-cluster-specific use cases (Scope, Change, Knowledge, etc.). |
| **Industry-Agnostic Framework** | Designed to fit any enterprise PPM landscape, regardless of tools or integration maturity. |

---

## Architecture Overview
User (Voice/Text)
│
▼
ReactNative UI → Streamlit Frontend
│
▼
Word Bank + Decision Tree (NLP Intent Layer)
│
▼
AI Engine (Python + Scikit-Learn + Rule-Based Predictor)
│
▼
Risk Score Model → Predictive Output → Agentic Response
│
▼
User Dashboard & Insights Visualization


 *Core Components*
- **app_streamlit.py** → Streamlit dashboard for visualization and Q&A  
- **baseline_risk_scoring.py** → Rule-based baseline scoring model  
- **train_model_10d.py** → ML training script using static dataset  
- **agent_cli.py** → Command-line AI assistant for textual queries  
- **tasks_static_10d.csv** → Static dataset representing project tasks  

---

## Setup & Usage

### Option 1 —Run Locally 
1. Click **Code ▸ Codespaces ▸ Create Codespace on main**.  
2. In the Codespaces terminal, run:
   ```bash
   pip install -r requirements.txt
   streamlit run app_streamlit.py --server.port 8501 --server.address 0.0.0.0
### Option 2 — Run in GitHub Codespaces *(recommended for demo)*
1. # Clone the repository 
git clone https://github.com/writetosoumya/RoboticPMO-Schedule_risk.git
cd RoboticPMO-Schedule_risk
2. # Create and activate a virtual environment:
python -m venv .venv
.\.venv\Scripts\activate        # (Windows)
# or
source .venv/bin/activate      # (Mac/Linux)
3. Install Dependencies
pip install -r requirements.txt
4. Run the app
streamlit run app_streamlit.py

##Future Roadmap
| Stream                     | Planned Enhancements                                            |
| -------------------------- | --------------------------------------------------------------- |
| **Scope Management**       | Predictive alerts on scope creep, requirement volatility        |
| **Change Management**      | AI-driven analysis of CR impact on delivery timelines           |
| **Knowledge Management**   | NLP-driven retrieval of historical lessons and mitigations      |
| **Integration Enablement** | Optional APIs to connect with Jira, MS Project, SAP, or Clarity |
| **Voice Augmentation**     | Integration with Salesforce Einstein / AgentForce for voice AI  |

# Vision

To transform traditional PMO governance into an AI-native ecosystem that continuously learns, predicts, and adapts — making enterprise transformation smarter, faster, and more transparent.

This framework is envisioned to evolve into a scalable Cognizant accelerator, enabling tailored AI-PMO solutions for diverse industries and tools.

#License
© 2025 Soumya Jom. All rights reserved.
This repository and its contents are proprietary intellectual property developed as part of the Robotic PMO – AI-Native Prototype framework.

Unauthorized reproduction, modification, or commercial use is prohibited without written permission from the copyright holder.

For partnership or licensing discussions:
Soumyamol Vijayamma Surendran | Cognizant Business Consulting
Linked in: https://www.linkedin.com/in/businessstrategistnyc/
⭐ If you found this project inspiring or innovative, consider giving it a star on GitHub! 🌟
