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