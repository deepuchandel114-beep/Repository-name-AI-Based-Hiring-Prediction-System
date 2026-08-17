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

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Logistic Regression

---

## 📊 Model Performance

| Metric | Result |
|---|---:|
| Training Samples | 800 |
| Testing Samples | 200 |
| Test Accuracy | **99.50%** |
| Model | Logistic Regression |

---

## 📌 Features Used

### Numerical Features

- Experience
- Salary Expectation
- Projects Count
- AI Score

### Categorical Features

- Skills
- Education
- Certifications
- Job Role

---

## 🔍 Feature Importance

The model analyzes which features have the strongest influence on hiring predictions.

The project includes:

- `feature_importance.csv`
- Top 10 Feature Importance visualization

---

## 📸 Project Screenshots
### Hiring Decision Distribution

![Hiring Decision Distribution](screenshots/Hiring%20decision%20distribution.png)

### AI Score vs Hiring Decision

![AI Score vs Hiring Decision](screenshots/AI%20Score%20vs%20Hiring%20Decision.png)

### Confusion Matrix

![Confusion Matrix](screenshots/Confusion%20Matrix.png)

### Top 10 Feature Importance

![Top 10 Feature Importance](screenshots/Top%2010%20Feature%20Importance.png)

### Candidate Prediction Demo

![Candidate Prediction Demo](screenshots/Candidate%20Prediction%20Demo.png)

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/deepuchandel114-beep/Repository-name-AI-Based-Hiring-Prediction-System.git