# 🤖 AI-Based Hiring Prediction System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

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
```

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

### Education vs Hiring Decision

![Education vs Hiring Decision](screenshots/Education%20vs%20Hiring%20decision.png)

### Job vs Hiring Decision
(![Job vs Hiring Decision](screenshots/Job%20vs%20Hiring%20Decision.png))

### Project Count vs Hiring Decision

![Project Count vs Hiring Decision](screenshots/Project%20Count%20vs%20Hiring%20decision.png)

### Salary Expectation vs Hiring Decision

![Salary Expectation vs Hiring Decision](screenshots/Salary%20Expectation%20vs%20Hiring%20Decision.png)

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/deepuchandel114-beep/Repository-name-AI-Based-Hiring-Prediction-System.git
cd Repository-name-AI-Based-Hiring-Prediction-System
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Project

```bash
python hiring_prediction.py
```

The system will train the model, evaluate its performance, save the trained model and encoder, perform candidate predictions, and generate result files.

---

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
```

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

---

## 👨‍💻 Author

**Deepak**

B.Tech — Artificial Intelligence

This project was developed as an end-to-end Machine Learning project focused on applying predictive analytics to recruitment and hiring decisions.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## ⚠️ Limitations

This project is intended for educational and demonstration purposes.

The prediction generated by the model should not be used as the sole basis for real-world hiring decisions. Actual recruitment decisions should consider human evaluation, interviews, qualifications, experience, organizational requirements, and applicable fairness and employment policies.

The model's performance also depends on the quality, size, and characteristics of the training dataset.

---
