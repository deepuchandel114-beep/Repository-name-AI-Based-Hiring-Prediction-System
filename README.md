# 🤖 AI-Based Hiring Prediction System

An AI-powered machine learning system that predicts whether a candidate is likely to be **Hired** or **Rejected** based on candidate-related features such as experience, skills, education, certifications, projects, salary expectation, and AI score.

---

## 📌 Project Overview

The system uses historical hiring data to train a Machine Learning model.

The trained model analyzes candidate information and predicts:

- HIRE
- REJECT

It also provides the probability of both outcomes.

---

## 🎯 Objectives

- Analyze historical hiring data
- Perform data cleaning and preprocessing
- Explore important hiring patterns
- Convert categorical data into numerical features
- Train a Machine Learning classification model
- Evaluate model performance
- Predict hiring decisions for new candidates
- Provide prediction probabilities
- Analyze important features influencing predictions

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

### Machine Learning

- One-Hot Encoding
- Logistic Regression
- Train-Test Split
- Classification Report
- Confusion Matrix

---

## 📂 Project Structure

```text
AI-Based-Hiring-Prediction-System/
│
├── Data/
│   └── hiring_data.csv
│
├── hiring_prediction.py
├── hiring_model.pkl
├── hiring_encoder.pkl
├── feature_importance.csv
├── prediction_result.csv
├── multiple_candidate_results.csv
├── requirements.txt
└── README.md
## 📊 Project Screenshots

### Hiring Decision Distribution

![Hiring Decision Distribution](screenshots/Hiring%20decision%20distribution.png)

### AI Score vs Hiring Decision

![AI Score vs Hiring Decision](screenshots/AI%20Score%20vs%20Hiring%20Decision.png)

### Confusion Matrix

![Confusion Matrix](screenshots/Confusion%20Matrix.png)

### Education vs Hiring Decision

![Education vs Hiring Decision](screenshots/Education%20vs%20Hiring%20decision.png)

### Job Role vs Hiring Decision

![Job Role vs Hiring Decision](screenshots/Job%20vs%20Hiring%20Decision.png)

### Projects Count vs Hiring Decision

![Projects Count vs Hiring Decision](screenshots/Project%20Count%20vs%20Hiring%20decision.png)

### Salary Expectation vs Hiring Decision

![Salary Expectation vs Hiring Decision](screenshots/Salary%20Expectation%20vs%20Hiring%20Decision%20.png)

### Top 10 Feature Importance

![Top 10 Feature Importance](screenshots/Top%2010%20Feature%20Importance.png)