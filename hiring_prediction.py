import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

df = pd.read_csv("Data/hiring_data.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nRecruiter Decision:")
print(df["Recruiter Decision"].value_counts())


# ============================================================
# STEP 2: DATA CLEANING
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())

df["Certifications"] = df["Certifications"].fillna(
    "No Certification"
)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)


# ============================================================
# STEP 3: HIRING DECISION DISTRIBUTION
# ============================================================

df["Recruiter Decision"].value_counts().plot(
    kind="bar"
)

plt.title("Hiring Decision Distribution")
plt.xlabel("Recruiter Decision")
plt.ylabel("Number of Candidates")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 4: EXPERIENCE VS HIRING
# ============================================================

sns.boxplot(
    x="Recruiter Decision",
    y="Experience (Years)",
    data=df
)

plt.title("Experience vs Hiring Decision")
plt.xlabel("Recruiter Decision")
plt.ylabel("Experience (Years)")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 5: AI SCORE VS HIRING
# ============================================================

sns.boxplot(
    x="Recruiter Decision",
    y="AI Score (0-100)",
    data=df
)

plt.title("AI Score vs Hiring Decision")
plt.xlabel("Recruiter Decision")
plt.ylabel("AI Score")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 6: PROJECTS VS HIRING
# ============================================================

sns.boxplot(
    x="Recruiter Decision",
    y="Projects Count",
    data=df
)

plt.title("Projects Count vs Hiring Decision")
plt.xlabel("Recruiter Decision")
plt.ylabel("Projects Count")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 7: SALARY VS HIRING
# ============================================================

sns.boxplot(
    x="Recruiter Decision",
    y="Salary Expectation ($)",
    data=df
)

plt.title("Salary Expectation vs Hiring Decision")
plt.xlabel("Recruiter Decision")
plt.ylabel("Salary Expectation ($)")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 8: EDUCATION VS HIRING
# ============================================================

pd.crosstab(
    df["Education"],
    df["Recruiter Decision"]
).plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Education vs Hiring Decision")
plt.xlabel("Education")
plt.ylabel("Number of Candidates")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# STEP 9: JOB ROLE VS HIRING
# ============================================================

pd.crosstab(
    df["Job Role"],
    df["Recruiter Decision"]
).plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Job Role vs Hiring Decision")
plt.xlabel("Job Role")
plt.ylabel("Number of Candidates")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# STEP 10: SKILL ANALYSIS
# ============================================================

skills = df["Skills"].str.split(", ").explode()

print("\nTop 10 Skills:")
print(
    skills.value_counts().head(10)
)


# ============================================================
# STEP 11: PYTHON SKILL VS HIRING
# ============================================================

python_candidates = df[
    df["Skills"].str.contains(
        "Python",
        na=False
    )
]

print("\nPython Skill vs Hiring:")
print(
    python_candidates[
        "Recruiter Decision"
    ].value_counts()
)


# ============================================================
# STEP 12: TOP SKILLS HIRING RATE
# ============================================================

top_skills = skills.value_counts().head(10).index

print("\nTop Skills Hiring Rate:")

for skill in top_skills:

    candidates = df[
        df["Skills"].str.contains(
            skill,
            na=False
        )
    ]

    hire_rate = (
        candidates["Recruiter Decision"]
        .eq("Hire")
        .mean()
        * 100
    )

    print(
        f"{skill}: {hire_rate:.2f}% hired"
    )


# ============================================================
# STEP 13: FEATURES AND TARGET
# ============================================================

X = df.drop(
    [
        "Recruiter Decision",
        "Resume_ID",
        "Name"
    ],
    axis=1
)

y = df["Recruiter Decision"]

print("\nFinal Features:")
print(X.columns)

print("\nTarget:")
print(y.name)


# ============================================================
# STEP 14: DEFINE COLUMNS
# ============================================================

categorical_columns = [
    "Skills",
    "Education",
    "Certifications",
    "Job Role"
]

numerical_columns = [
    "Experience (Years)",
    "Salary Expectation ($)",
    "Projects Count",
    "AI Score (0-100)"
]


# ============================================================
# STEP 15: ONE-HOT ENCODING
# ============================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded_data = encoder.fit_transform(
    X[categorical_columns]
)

print("\nEncoded Data Shape:")
print(encoded_data.shape)


# ============================================================
# STEP 16: CREATE FINAL DATASET
# ============================================================

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(
        categorical_columns
    )
)

numerical_df = X[
    numerical_columns
].reset_index(drop=True)

X_final = pd.concat(
    [
        numerical_df,
        encoded_df
    ],
    axis=1
)

print("\nFinal Dataset Shape:")
print(X_final.shape)

print("\nFirst 5 Rows:")
print(X_final.head())


# ============================================================
# STEP 17: TARGET ENCODING
# ============================================================

y_final = y.map({
    "Reject": 0,
    "Hire": 1
})

print("\nTarget Encoding:")
print(y_final.value_counts())


# ============================================================
# STEP 18: TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y_final,
    test_size=0.2,
    random_state=42,
    stratify=y_final
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ============================================================
# STEP 19: CREATE AND TRAIN MODEL
# ============================================================

model = LogisticRegression(
    max_iter=2000
)

model.fit(
    X_train,
    y_train
)

print("\nModel Training Complete!")
print("Model is ready for prediction!")


# ============================================================
# STEP 20: TEST DATA PREDICTION
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)

print("\nPrediction Probabilities:")
print(y_prob[:5])

print("\nPredictions:")
print(y_pred[:20])


# ============================================================
# STEP 21: ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)


# ============================================================
# STEP 22: CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Reject",
            "Hire"
        ]
    )
)


# ============================================================
# STEP 23: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "Reject",
        "Hire"
    ],
    yticklabels=[
        "Reject",
        "Hire"
    ]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 24: SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "hiring_model.pkl"
)

print("\nModel saved successfully!")


# ============================================================
# STEP 25: SAVE ENCODER
# ============================================================

joblib.dump(
    encoder,
    "hiring_encoder.pkl"
)

print("Encoder saved successfully!")


# ============================================================
# STEP 26: LOAD MODEL AND ENCODER
# ============================================================

loaded_model = joblib.load(
    "hiring_model.pkl"
)

loaded_encoder = joblib.load(
    "hiring_encoder.pkl"
)

print("\nModel loaded successfully!")
print("Encoder loaded successfully!")


# ============================================================
# STEP 27: INTEGER INPUT VALIDATION
# ============================================================

def get_integer(
    prompt,
    minimum=0,
    maximum=None
):

    while True:

        try:

            value = int(
                input(prompt)
            )

            if value < minimum:

                print(
                    f"Value must be at least {minimum}."
                )

                continue

            if (
                maximum is not None
                and value > maximum
            ):

                print(
                    f"Value must be at most {maximum}."
                )

                continue

            return value

        except ValueError:

            print(
                "Please enter a valid number."
            )


# ============================================================
# STEP 28: USER INPUT
# ============================================================

print("\n")
print("=" * 55)
print("          AI HIRING PREDICTION SYSTEM")
print("=" * 55)


experience = get_integer(
    "Enter Experience (Years): ",
    0,
    50
)

salary = get_integer(
    "Enter Salary Expectation ($): ",
    0
)

projects = get_integer(
    "Enter Projects Count: ",
    0,
    100
)

ai_score = get_integer(
    "Enter AI Score (0-100): ",
    0,
    100
)

skills = input(
    "Enter Skills: "
).strip()

education = input(
    "Enter Education: "
).strip()

certification = input(
    "Enter Certification: "
).strip()

job_role = input(
    "Enter Job Role: "
).strip()


# ============================================================
# STEP 29: CREATE CANDIDATE DATAFRAME
# ============================================================

candidate = pd.DataFrame({

    "Skills": [skills],

    "Experience (Years)": [
        experience
    ],

    "Education": [
        education
    ],

    "Certifications": [
        certification
    ],

    "Job Role": [
        job_role
    ],

    "Salary Expectation ($)": [
        salary
    ],

    "Projects Count": [
        projects
    ],

    "AI Score (0-100)": [
        ai_score
    ]
})


# ============================================================
# STEP 30: ENCODE CANDIDATE
# ============================================================

candidate_encoded = loaded_encoder.transform(
    candidate[categorical_columns]
)

candidate_encoded_df = pd.DataFrame(
    candidate_encoded,
    columns=loaded_encoder.get_feature_names_out(
        categorical_columns
    )
)


# ============================================================
# STEP 31: NUMERICAL FEATURES
# ============================================================

candidate_numerical = candidate[
    numerical_columns
].reset_index(drop=True)


# ============================================================
# STEP 32: FINAL CANDIDATE FEATURES
# ============================================================

candidate_final = pd.concat(
    [
        candidate_numerical,
        candidate_encoded_df
    ],
    axis=1
)

print("\nCandidate Feature Shape:")
print(candidate_final.shape)


# ============================================================
# STEP 33: MAKE PREDICTION
# ============================================================

prediction = loaded_model.predict(
    candidate_final
)

prediction_probability = (
    loaded_model.predict_proba(
        candidate_final
    )
)


# ============================================================
# STEP 34: GET PROBABILITIES
# ============================================================

reject_probability = (
    prediction_probability[0][0] * 100
)

hire_probability = (
    prediction_probability[0][1] * 100
)


# ============================================================
# STEP 35: DETERMINE DECISION
# ============================================================

if prediction[0] == 1:

    decision = "HIRE"

else:

    decision = "REJECT"


# ============================================================
# STEP 36: FINAL RESULT
# ============================================================

print("\n")
print("=" * 55)
print("             AI HIRING PREDICTION")
print("=" * 55)

print("\nCandidate Information")
print("-" * 55)

print(
    f"Experience        : {experience} years"
)

print(
    f"Salary Expectation: ${salary}"
)

print(
    f"Projects          : {projects}"
)

print(
    f"AI Score          : {ai_score}/100"
)

print(
    f"Skills            : {skills}"
)

print(
    f"Education         : {education}"
)

print(
    f"Certification     : {certification}"
)

print(
    f"Job Role          : {job_role}"
)

print("\nPrediction Result")
print("-" * 55)

print(
    f"Decision          : {decision}"
)

print(
    f"Hire Probability  : {hire_probability:.2f}%"
)

print(
    f"Reject Probability: {reject_probability:.2f}%"
)

print("=" * 55)
# ============================================================
# STEP 47: FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X_final.columns,
    "Coefficient": loaded_model.coef_[0]
})

# Absolute importance
feature_importance["Importance"] = (
    feature_importance["Coefficient"].abs()
)

# Sort by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 55)
print("          TOP 10 IMPORTANT FEATURES")
print("=" * 55)

print(
    feature_importance[
        ["Feature", "Coefficient"]
    ].head(10)
)
# ============================================================
# STEP 48: FEATURE IMPORTANCE VISUALIZATION
# ============================================================

# ============================================================
# STEP 48: FEATURE IMPORTANCE VISUALIZATION
# ============================================================

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)

plt.title("Top 10 Feature Importance")
plt.xlabel("Coefficient")
plt.ylabel("Feature")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
# ============================================================
# STEP 49: SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")
print("File: feature_importance.csv")
# ============================================================
# STEP 50: SAVE CANDIDATE PREDICTION
# ============================================================

prediction_result = pd.DataFrame({
    "Experience (Years)": [experience],
    "Salary Expectation ($)": [salary],
    "Projects Count": [projects],
    "AI Score (0-100)": [ai_score],
    "Skills": [skills],
    "Education": [education],
    "Certification": [certification],
    "Job Role": [job_role],
    "Decision": [decision],
    "Hire Probability (%)": [round(hire_probability, 2)],
    "Reject Probability (%)": [round(reject_probability, 2)]
})

prediction_result.to_csv(
    "prediction_result.csv",
    index=False
)

print("\nPrediction result saved successfully!")
print("File: prediction_result.csv")
# ============================================================
# STEP 51: MULTIPLE CANDIDATE PREDICTION
# ============================================================

print("\n")
print("=" * 60)
print("        MULTIPLE CANDIDATE PREDICTION")
print("=" * 60)

# Sample candidates
test_candidates = pd.DataFrame({

    "Skills": [
        "Python, Machine Learning",
        "Java, SQL",
        "Python, Deep Learning"
    ],

    "Experience (Years)": [
        6,
        2,
        8
    ],

    "Education": [
        "B.Tech",
        "B.Sc",
        "M.Tech"
    ],

    "Certifications": [
        "AWS Certified",
        "No Certification",
        "TensorFlow Certified"
    ],

    "Job Role": [
        "Data Scientist",
        "Software Engineer",
        "AI Researcher"
    ],

    "Salary Expectation ($)": [
        85000,
        60000,
        95000
    ],

    "Projects Count": [
        5,
        2,
        7
    ],

    "AI Score (0-100)": [
        90,
        55,
        95
    ]
})


# Encode categorical features
test_encoded = loaded_encoder.transform(
    test_candidates[categorical_columns]
)


# Convert encoded features to DataFrame
test_encoded_df = pd.DataFrame(
    test_encoded,
    columns=loaded_encoder.get_feature_names_out(
        categorical_columns
    )
)


# Numerical features
test_numerical = test_candidates[
    numerical_columns
].reset_index(drop=True)


# Combine numerical + encoded features
test_final = pd.concat(
    [
        test_numerical,
        test_encoded_df
    ],
    axis=1
)


# Make predictions
test_predictions = loaded_model.predict(
    test_final
)

test_probabilities = loaded_model.predict_proba(
    test_final
)


# Add results
test_candidates["Prediction"] = [
    "HIRE" if prediction == 1 else "REJECT"
    for prediction in test_predictions
]

test_candidates["Hire Probability (%)"] = (
    test_probabilities[:, 1] * 100
).round(2)

test_candidates["Reject Probability (%)"] = (
    test_probabilities[:, 0] * 100
).round(2)


# Display results
print("\nMultiple Candidate Results:")
print("-" * 60)

print(
    test_candidates[
        [
            "Job Role",
            "Experience (Years)",
            "AI Score (0-100)",
            "Prediction",
            "Hire Probability (%)",
            "Reject Probability (%)"
        ]
    ].to_string(index=False)
)# ============================================================
# STEP 52: SAVE MULTIPLE CANDIDATE RESULTS
# ============================================================

test_candidates.to_csv(
    "multiple_candidate_results.csv",
    index=False
)

print("\nMultiple candidate results saved successfully!")
print("File: multiple_candidate_results.csv")
# ============================================================
# STEP 53: FINAL PROJECT SUMMARY
# ============================================================

print("\n")
print("=" * 65)
print("              AI-BASED HIRING PREDICTION SYSTEM")
print("=" * 65)

print("\nPROJECT STATUS")
print("-" * 65)

print("Dataset Loaded              : YES")
print("Data Cleaning Completed     : YES")
print("Feature Engineering         : YES")
print("One-Hot Encoding            : YES")
print("Model Training              : COMPLETED")
print("Model Evaluation            : COMPLETED")
print("Model Saved                 : hiring_model.pkl")
print("Encoder Saved               : hiring_encoder.pkl")
print("Feature Analysis            : COMPLETED")
print("Candidate Prediction        : AVAILABLE")
print("Multiple Candidate Testing  : AVAILABLE")

print("\nMODEL PERFORMANCE")
print("-" * 65)

print(
    f"Test Accuracy               : {accuracy * 100:.2f}%"
)

print(
    f"Test Samples                : {len(y_test)}"
)

print(
    f"Training Samples            : {len(y_train)}"
)

print("\nOUTPUT FILES")
print("-" * 65)

print("1. hiring_model.pkl")
print("2. hiring_encoder.pkl")
print("3. feature_importance.csv")
print("4. prediction_result.csv")
print("5. multiple_candidate_results.csv")

print("\n" + "=" * 65)
print("              SYSTEM READY FOR DEMONSTRATION")
print("=" * 65)