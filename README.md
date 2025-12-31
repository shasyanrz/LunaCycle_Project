# 🌙 LunaCycle 🌸
AI-Powered Biomedical Companion for Menstrual Phase Prediction and Fertility Insights

LunaCycle is a biomedical AI system designed to predict menstrual cycle phases and provide fertility-related insights using a combination of classical machine learning models and Large Language Model (LLM)–assisted interpretation. The system emphasizes explainability, reproducibility, and clear separation between predictive computation and natural language reasoning.

# 🔍 Project Overview

Menstrual cycle tracking applications often rely on simple calendar-based heuristics that fail to capture individual variability and provide limited interpretability for non-technical users. LunaCycle addresses this limitation by integrating a deterministic machine learning pipeline for phase prediction with an AI-assisted explanation layer that translates numerical outputs into contextual, user-friendly insights.

The system is developed as part of an Artificial Intelligence (Biomedical) course project and follows a research-to-production mindset, ensuring that both the modeling and system architecture remain extensible and reproducible.

# 🔑 Key Features

* **Menstrual Phase Prediction**
Predicts four menstrual cycle phases: Menstrual, Follicular, Ovulation, and Luteal.

* **Hormone Dynamics Visualization**
Generates interpretable hormone curves (Estrogen, Progesterone, LH, FSH) using deterministic mathematical modeling grounded in biomedical literature.

* **AI Insight Module**
Produces structured health and fertility summaries based on machine learning outputs using an LLM reasoning layer.

* **Context-Aware Chatbot**
Allows users to ask natural language questions about their cycle, with responses grounded in their latest prediction context.

# ⚙️ System Architecture

<img width="710" height="403" alt="image" src="https://github.com/user-attachments/assets/a1ceb01a-fcf7-4609-904f-944a5a911a20" />

# 🗺️ Repository Structure

```text
LunaCycle_Project/
├── backend/
│   ├── models/                # Trained ML model and scaler
│   ├── server.py              # FastAPI backend (API, ML inference, LLM handler)
│   ├── requirements.txt       # Backend dependencies
│   └── .env                   # Environment variables (ignored in git)
│
├── frontend/
│   ├── assets/                # Images and icons
│   ├── css/
│   │   └── style.css          # Global styling
│   ├── index.html             # Home page
│   ├── lunacycle.html         # Prediction & visualization
│   └── chatbot.html           # Context-aware chatbot
│
├── .gitignore
└── README.md
```






