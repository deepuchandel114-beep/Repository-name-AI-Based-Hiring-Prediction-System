# 🤖 AI-Based Hiring Prediction System

An end-to-end Machine Learning system that predicts whether a candidate should be **Hired or Rejected** based on their experience, skills, education, certifications, projects, salary expectation, and AI score.

---

## 🎯 Project Objective

The goal of this project is to assist recruiters in making data-driven hiring decisions using Machine Learning.

The system provides:

- ✅ Hire / Reject prediction
- 📊 Hiring probability
- 📈 Model performance evaluation
- 🔍 Feature importance analysis
- 👥 Multiple candidate prediction
- 📁 Prediction result export

---

## 🧠 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
One-Hot Encoding
   ↓
Train-Test Split
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Candidate Prediction

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/deepuchandel114-beep/Repository-name-AI-Based-Hiring-Prediction-System.git
cd Repository-name-AI-Based-Hiring-Prediction-System

## 📂 Project Structure

```text
AI-Based-Hiring-Prediction-System/
│
├── Data/
│   └── hiring_data.csv
│
├── screenshots/
│   ├── AI Score vs Hiring Decision.png
│   ├── Candidate Prediction Demo.png
│   ├── Confusion Matrix.png
│   ├── Education vs Hiring decision.png
│   ├── Hiring decision distribution.png
│   ├── Job vs Hiring Decision.png
│   ├── Project Count vs Hiring decision.png
│   ├── Salary Expectation vs Hiring Decision.png
│   └── Top 10 Feature Importance.png
│
├── hiring_prediction.py
├── hiring_model.pkl
├── hiring_encoder.pkl
├── feature_importance.csv
├── prediction_result.csv
├── multiple_candidate_results.csv
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore

---

## 🚀 Future Improvements

The current system can be further enhanced with the following features:

- Build an interactive web interface using Streamlit or Flask
- Allow recruiters to enter candidate details through a form
- Add resume parsing and automatic candidate information extraction
- Integrate NLP-based skill matching
- Compare multiple Machine Learning algorithms
- Add model explainability using SHAP
- Store candidate predictions in a database
- Deploy the application on a cloud platform
- Add recruiter authentication and candidate management
- Develop a REST API for real-time hiring predictions
