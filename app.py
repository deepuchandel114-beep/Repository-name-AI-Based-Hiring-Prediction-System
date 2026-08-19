import re
import pdfplumber
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Hiring Prediction System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "resume_id" not in st.session_state:
    st.session_state.resume_id = None

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "resume_info" not in st.session_state:
    st.session_state.resume_info = None


# ============================================================
# LOAD MODEL AND ENCODER
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("hiring_model.pkl")
    encoder = joblib.load("hiring_encoder.pkl")
    return model, encoder


try:
    model, encoder = load_model()
except Exception as error:
    st.error("Unable to load the trained model or encoder.")
    st.exception(error)
    st.stop()


# ============================================================
# RESUME INFORMATION EXTRACTION
# ============================================================

def extract_resume_info(text):
    text_lower = text.lower()

    # --------------------------------------------------------
    # EXPERIENCE
    # --------------------------------------------------------

    experience = 2

    experience_patterns = [
        r"(\d+)\s*\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)",
        r"(?:experience|exp)\s*[:\-]?\s*(\d+)\s*\+?\s*(?:years?|yrs?)"
    ]

    for pattern in experience_patterns:
        match = re.search(pattern, text_lower)
        if match:
            experience = int(match.group(1))
            break

    experience = max(0, min(experience, 50))

    # --------------------------------------------------------
    # SKILLS
    # --------------------------------------------------------

    possible_skills = [
        "python",
        "java",
        "c++",
        "c",
        "javascript",
        "typescript",
        "react",
        "node.js",
        "node",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "tensorflow",
        "pytorch",
        "keras",
        "scikit-learn",
        "sklearn",
        "sql",
        "mysql",
        "mongodb",
        "django",
        "flask",
        "fastapi",
        "aws",
        "azure",
        "docker",
        "git",
        "github",
        "html",
        "css",
        "pandas",
        "numpy"
    ]

    detected_skills = []

    for skill in possible_skills:
        if skill in text_lower:
            if skill == "sklearn":
                display_skill = "Scikit-learn"
            elif skill == "node":
                display_skill = "Node.js"
            elif skill == "artificial intelligence":
                display_skill = "Artificial Intelligence"
            else:
                display_skill = skill

            if display_skill not in detected_skills:
                detected_skills.append(display_skill)

    skills = (
        ", ".join(detected_skills)
        if detected_skills
        else "Python, Machine Learning"
    )

    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    education = "B.Sc"

    if (
        "b.tech" in text_lower
        or "btech" in text_lower
        or "bachelor of technology" in text_lower
    ):
        education = "B.Tech"

    elif (
        "m.tech" in text_lower
        or "mtech" in text_lower
        or "master of technology" in text_lower
    ):
        education = "M.Tech"

    elif "mba" in text_lower:
        education = "MBA"

    elif "phd" in text_lower or "doctorate" in text_lower:
        education = "PhD"

    elif (
        "b.sc" in text_lower
        or "bsc" in text_lower
        or "bachelor of science" in text_lower
    ):
        education = "B.Sc"

    # --------------------------------------------------------
    # CERTIFICATION
    # --------------------------------------------------------

    certification = "No Certification"

    certification_keywords = [
        "certified",
        "certification",
        "certificate",
        "aws certified",
        "google certified",
        "microsoft certified"
    ]

    if any(keyword in text_lower for keyword in certification_keywords):
        certification = "Certification Found"

    # --------------------------------------------------------
    # JOB ROLE
    # --------------------------------------------------------

    job_role = "Software Engineer"

    job_roles = {
        "data scientist": "Data Scientist",
        "data analyst": "Data Analyst",
        "machine learning engineer": "Machine Learning Engineer",
        "ai engineer": "AI Engineer",
        "artificial intelligence engineer": "AI Engineer",
        "software engineer": "Software Engineer",
        "software developer": "Software Developer",
        "web developer": "Web Developer",
        "full stack developer": "Full Stack Developer",
        "frontend developer": "Frontend Developer",
        "backend developer": "Backend Developer",
        "python developer": "Python Developer",
        "devops engineer": "DevOps Engineer",
        "data engineer": "Data Engineer"
    }

    for role_keyword, role_name in job_roles.items():
        if role_keyword in text_lower:
            job_role = role_name
            break

    # --------------------------------------------------------
    # PROJECT COUNT
    # --------------------------------------------------------

    projects = 3

    project_patterns = [
        r"(\d+)\s*(?:\+)?\s*projects?",
        r"projects?\s*[:\-]?\s*(\d+)"
    ]

    for pattern in project_patterns:
        match = re.search(pattern, text_lower)
        if match:
            projects = int(match.group(1))
            break

    projects = max(0, min(projects, 100))

    return {
        "experience": experience,
        "skills": skills,
        "education": education,
        "certification": certification,
        "job_role": job_role,
        "projects": projects
    }


# ============================================================
# RESUME SCORE
# ============================================================

def calculate_resume_score(
    experience,
    projects,
    skills,
    education,
    certification
):
    score = 0

    # Experience: 25 points
    score += min(experience * 5, 25)

    # Projects: 20 points
    score += min(projects * 4, 20)

    # Skills: 25 points
    skill_count = len(
        [item for item in skills.split(",") if item.strip()]
    )
    score += min(skill_count * 5, 25)

    # Education: 20 points
    education_scores = {
        "B.Sc": 10,
        "B.Tech": 15,
        "M.Tech": 18,
        "MBA": 15,
        "PhD": 20
    }
    score += education_scores.get(education, 10)

    # Certification: 10 points
    if certification != "No Certification":
        score += 10

    return min(int(score), 100)


# ============================================================
# HELPER: SAFE ENCODING
# ============================================================

def encode_candidate(candidate, categorical_columns):
    """
    Uses the saved encoder. If the encoder does not support
    unknown categories, unseen values are converted to zero
    encoded features where possible.
    """

    try:
        encoded_data = encoder.transform(
            candidate[categorical_columns]
        )

    except Exception:
        # Compatible fallback for a saved OneHotEncoder.
        if not hasattr(encoder, "categories_"):
            raise

        encoded_values = []

        for column in categorical_columns:
            value = candidate[column].iloc[0]
            categories = list(
                encoder.categories_[
                    categorical_columns.index(column)
                ]
            )

            encoded_values.append(
                1 if value in categories else 0
            )

        # Build using encoder's normal feature names.
        feature_names = encoder.get_feature_names_out(
            categorical_columns
        )

        # Recreate a zero vector and activate the matching
        # category for each column.
        encoded_vector = [0] * len(feature_names)

        offset = 0

        for column, value in zip(
            categorical_columns,
            candidate[categorical_columns].iloc[0]
        ):
            categories = list(
                encoder.categories_[
                    categorical_columns.index(column)
                ]
            )

            for category_index, category in enumerate(categories):
                if value == category:
                    encoded_vector[offset + category_index] = 1
                    break

            offset += len(categories)

        encoded_data = [encoded_vector]

    return pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(
            categorical_columns
        )
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🤖 AI Hiring System")
    st.divider()

    st.subheader("🧠 Model Information")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Prediction:** Hire / Reject")
    st.write("**Input:** Resume + Candidate Features")

    st.divider()

    st.subheader("📊 Prediction Features")
    st.write("• Experience")
    st.write("• Salary Expectation")
    st.write("• Projects")
    st.write("• AI Assessment Score")
    st.write("• Skills")
    st.write("• Education")
    st.write("• Certifications")
    st.write("• Job Role")

    st.divider()

    if st.session_state.prediction_history:
        if st.button(
            "🗑️ Clear Prediction History",
            use_container_width=True
        ):
            st.session_state.prediction_history = []
            st.rerun()

    st.caption("AI-Based Hiring Prediction System")
    st.caption("Machine Learning Internship Project")


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🤖 AI-Based Hiring Prediction System")

st.write(
    "An end-to-end Machine Learning application that analyzes "
    "candidate information and predicts a hiring decision."
)

st.divider()


# ============================================================
# DASHBOARD
# ============================================================

history_df = pd.DataFrame(
    st.session_state.prediction_history
)

if not history_df.empty:
    total_candidates = len(history_df)
    total_hired = history_df["Decision"].eq("Hire").sum()
    total_rejected = history_df["Decision"].eq("Reject").sum()
    hiring_rate = total_hired / total_candidates * 100
else:
    total_candidates = 0
    total_hired = 0
    total_rejected = 0
    hiring_rate = 0

st.subheader("📊 Hiring Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👥 Total Candidates", total_candidates)

with col2:
    st.metric("✅ Hired", total_hired)

with col3:
    st.metric("❌ Rejected", total_rejected)

with col4:
    st.metric("📈 Hiring Rate", f"{hiring_rate:.2f}%")

st.divider()


# ============================================================
# RESUME UPLOAD
# ============================================================

st.header("📄 Resume Analysis")

uploaded_resume = st.file_uploader(
    "Upload Candidate Resume (PDF)",
    type=["pdf"],
    key="resume_uploader"
)

if uploaded_resume is not None:

    current_resume_id = (
        uploaded_resume.name,
        uploaded_resume.size
    )

    if st.session_state.resume_id != current_resume_id:

        extracted_text = ""

        try:
            with pdfplumber.open(uploaded_resume) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()

                    if page_text:
                        extracted_text += page_text + "\n"

        except Exception as error:
            st.error("Unable to read this PDF.")
            st.exception(error)
            st.stop()

        st.session_state.resume_text = extracted_text
        st.session_state.resume_id = current_resume_id

        if extracted_text.strip():
            st.session_state.resume_info = extract_resume_info(
                extracted_text
            )
        else:
            st.session_state.resume_info = None

    if st.session_state.resume_text.strip():

        st.success(
            f"✅ Resume analyzed: {uploaded_resume.name}"
        )

        with st.expander("📄 View Extracted Resume Text"):
            st.text_area(
                "Resume Content",
                st.session_state.resume_text,
                height=250,
                disabled=True
            )

        resume_info = st.session_state.resume_info

        resume_score = calculate_resume_score(
            resume_info["experience"],
            resume_info["projects"],
            resume_info["skills"],
            resume_info["education"],
            resume_info["certification"]
        )

        st.subheader("⭐ Resume Assessment")

        score_col1, score_col2 = st.columns([1, 2])

        with score_col1:
            st.metric(
                "Resume Score",
                f"{resume_score}/100"
            )

        with score_col2:
            if resume_score >= 80:
                st.success("Strong Candidate Profile")
            elif resume_score >= 60:
                st.warning("Average Candidate Profile")
            else:
                st.info("Candidate Profile Needs Improvement")

        st.progress(resume_score)

        st.subheader("🤖 Extracted Candidate Information")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.write(
                f"**Experience:** {resume_info['experience']} years"
            )
            st.write(
                f"**Skills:** {resume_info['skills']}"
            )
            st.write(
                f"**Education:** {resume_info['education']}"
            )

        with info_col2:
            st.write(
                f"**Certification:** {resume_info['certification']}"
            )
            st.write(
                f"**Job Role:** {resume_info['job_role']}"
            )
            st.write(
                f"**Projects:** {resume_info['projects']}"
            )

    else:
        st.warning(
            "The PDF was uploaded, but no readable text was found. "
            "Please use a text-based PDF."
        )


# ============================================================
# CANDIDATE INFORMATION
# ============================================================

st.header("📋 Candidate Information")

default_experience = 2
default_projects = 3
default_skills = "Python, Machine Learning"
default_education = "B.Sc"
default_certification = "No Certification"
default_job_role = "Software Engineer"
default_ai_score = 70

if st.session_state.resume_info is not None:

    resume_info = st.session_state.resume_info

    default_experience = resume_info["experience"]
    default_projects = resume_info["projects"]
    default_skills = resume_info["skills"]
    default_education = resume_info["education"]
    default_certification = resume_info["certification"]
    default_job_role = resume_info["job_role"]

    default_ai_score = calculate_resume_score(
        resume_info["experience"],
        resume_info["projects"],
        resume_info["skills"],
        resume_info["education"],
        resume_info["certification"]
    )


col1, col2 = st.columns(2)

with col1:

    experience = st.number_input(
        "Experience (Years)",
        min_value=0,
        max_value=50,
        value=int(default_experience),
        step=1
    )

    salary = st.number_input(
        "Salary Expectation ($)",
        min_value=0,
        value=50000,
        step=1000
    )

    projects = st.number_input(
        "Projects Count",
        min_value=0,
        max_value=100,
        value=int(default_projects),
        step=1
    )

    ai_score = st.slider(
        "AI Assessment Score (0-100)",
        min_value=0,
        max_value=100,
        value=int(default_ai_score)
    )


with col2:

    skills = st.text_input(
        "Skills",
        value=default_skills
    )

    education_options = [
        "B.Sc",
        "B.Tech",
        "M.Tech",
        "MBA",
        "PhD"
    ]

    if default_education not in education_options:
        default_education = "B.Sc"

    education = st.selectbox(
        "Education",
        education_options,
        index=education_options.index(default_education)
    )

    certification = st.text_input(
        "Certification",
        value=default_certification
    )

    job_role = st.text_input(
        "Job Role",
        value=default_job_role
    )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

predict_button = st.button(
    "🔮 Predict Hiring Decision",
    use_container_width=True,
    type="primary"
)

if predict_button:

    candidate = pd.DataFrame({
        "Skills": [skills],
        "Experience (Years)": [experience],
        "Education": [education],
        "Certifications": [certification],
        "Job Role": [job_role],
        "Salary Expectation ($)": [salary],
        "Projects Count": [projects],
        "AI Score (0-100)": [ai_score]
    })

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

    try:

        encoded_df = encode_candidate(
            candidate,
            categorical_columns
        )

        numerical_df = candidate[
            numerical_columns
        ].reset_index(drop=True)

        final_features = pd.concat(
            [numerical_df, encoded_df],
            axis=1
        )

        # Match the exact feature order used during training.
        if hasattr(model, "feature_names_in_"):
            final_features = final_features.reindex(
                columns=list(model.feature_names_in_),
                fill_value=0
            )

        prediction = model.predict(final_features)
        probabilities = model.predict_proba(final_features)

    except Exception as error:

        st.error(
            "Prediction could not be completed with the current "
            "candidate values."
        )
        st.exception(error)
        st.stop()

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    classes = list(model.classes_)

    hire_probability = 0.0
    reject_probability = 0.0

    if 1 in classes:
        hire_probability = (
            probabilities[0][classes.index(1)] * 100
        )

    if 0 in classes:
        reject_probability = (
            probabilities[0][classes.index(0)] * 100
        )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    decision = "Hire" if prediction[0] == 1 else "Reject"

    resume_score_for_history = 0

    if st.session_state.resume_info is not None:
        resume_info = st.session_state.resume_info

        resume_score_for_history = calculate_resume_score(
            resume_info["experience"],
            resume_info["projects"],
            resume_info["skills"],
            resume_info["education"],
            resume_info["certification"]
        )

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_record = {
        "Experience": experience,
        "Salary": salary,
        "Projects": projects,
        "AI Score": ai_score,
        "Resume Score": resume_score_for_history,
        "Skills": skills,
        "Education": education,
        "Certification": certification,
        "Job Role": job_role,
        "Decision": decision,
        "Hire Probability": round(hire_probability, 2),
        "Reject Probability": round(reject_probability, 2)
    }

    st.session_state.prediction_history.append(
        history_record
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.divider()
    st.header("🎯 Prediction Result")

    if decision == "Hire":
        st.success(
            "🎉 CANDIDATE RECOMMENDED FOR HIRING"
        )
    else:
        st.error(
            "❌ CANDIDATE NOT RECOMMENDED FOR HIRING"
        )

    probability_col1, probability_col2 = st.columns(2)

    with probability_col1:
        st.metric(
            "✅ Hire Probability",
            f"{hire_probability:.2f}%"
        )

    with probability_col2:
        st.metric(
            "❌ Reject Probability",
            f"{reject_probability:.2f}%"
        )

    st.subheader("📊 Hiring Probability")

    st.progress(
        max(0, min(int(hire_probability), 100))
    )

    st.write(
        f"**{hire_probability:.2f}% probability of hiring**"
    )

    # --------------------------------------------------------
    # CANDIDATE SUMMARY
    # --------------------------------------------------------

    st.subheader("👤 Candidate Summary")

    summary_rows = [
        ["Experience", f"{experience} years"],
        ["Salary Expectation", f"${salary:,}"],
        ["Projects", projects],
        ["AI Assessment Score", f"{ai_score}/100"],
        ["Resume Score", f"{resume_score_for_history}/100"],
        ["Skills", skills],
        ["Education", education],
        ["Certification", certification],
        ["Job Role", job_role],
        ["Final Decision", decision]
    ]

    summary = pd.DataFrame(
        summary_rows,
        columns=["Feature", "Value"]
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DOWNLOAD RESULT
    # --------------------------------------------------------

    st.subheader("📥 Export Prediction")

    result_data = pd.DataFrame({
        "Experience": [experience],
        "Salary Expectation": [salary],
        "Projects": [projects],
        "AI Score": [ai_score],
        "Resume Score": [resume_score_for_history],
        "Skills": [skills],
        "Education": [education],
        "Certification": [certification],
        "Job Role": [job_role],
        "Prediction": [decision],
        "Hire Probability": [
            f"{hire_probability:.2f}%"
        ],
        "Reject Probability": [
            f"{reject_probability:.2f}%"
        ]
    })

    csv_data = result_data.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv_data,
        file_name="candidate_prediction.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.divider()
st.header("📚 Prediction History")

history_df = pd.DataFrame(
    st.session_state.prediction_history
)

if not history_df.empty:

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    history_csv = history_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction History",
        data=history_csv,
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.subheader("📈 Hiring Statistics")

    total_candidates = len(history_df)

    total_hired = (
        history_df["Decision"]
        .eq("Hire")
        .sum()
    )

    total_rejected = (
        history_df["Decision"]
        .eq("Reject")
        .sum()
    )

    hiring_rate = (
        total_hired / total_candidates * 100
    )

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

    with stats_col1:
        st.metric("Total", total_candidates)

    with stats_col2:
        st.metric("Hired", total_hired)

    with stats_col3:
        st.metric("Rejected", total_rejected)

    with stats_col4:
        st.metric("Hiring Rate", f"{hiring_rate:.2f}%")

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    st.subheader("📊 Hiring Decision Distribution")

    chart_data = pd.DataFrame({
        "Decision": ["Hired", "Rejected"],
        "Candidates": [total_hired, total_rejected]
    })

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        chart_data["Decision"],
        chart_data["Candidates"]
    )

    ax.set_xlabel("Hiring Decision")
    ax.set_ylabel("Number of Candidates")
    ax.set_title("Hiring Decision Distribution")

    st.pyplot(fig)
    plt.close(fig)

else:

    st.info(
        "No predictions yet. Make a prediction to see "
        "candidate history and statistics here."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🤖 AI-Based Hiring Prediction System | "
    "Machine Learning Internship Project"
)