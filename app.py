import re
import pdfplumber
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Hiring Prediction System", page_icon="🤖", layout="wide")

# ---------------- SESSION STATE ----------------
DEFAULTS = {
    "prediction_history": [],
    "resume_processed": None,
    "resume_text": "",
    "resume_info": None,
    "experience": 2,
    "projects": 3,
    "skills": "Python, Machine Learning",
    "education": "B.Sc",
    "certification": "No Certification",
    "job_role": "Software Engineer",
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("hiring_model.pkl"), joblib.load("hiring_encoder.pkl")

try:
    model, encoder = load_model()
except Exception as error:
    st.error("❌ Could not load hiring_model.pkl or hiring_encoder.pkl")
    st.exception(error)
    st.stop()

# ---------------- RESUME EXTRACTION ----------------
def extract_resume_info(text):
    t = text.lower()

    experience = 2
    patterns = [
        r"(\d+)\s*\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)",
        r"(?:experience|exp)\s*[:\-]?\s*(\d+)\s*\+?\s*(?:years?|yrs?)"
    ]
    for pattern in patterns:
        m = re.search(pattern, t)
        if m:
            experience = int(m.group(1))
            break
    experience = max(0, min(experience, 50))

    skill_names = [
        "python", "java", "c++", "c", "javascript", "typescript", "react",
        "node.js", "node", "machine learning", "deep learning",
        "artificial intelligence", "tensorflow", "pytorch", "keras",
        "scikit-learn", "sklearn", "sql", "mysql", "mongodb", "django",
        "flask", "fastapi", "aws", "azure", "docker", "git", "github",
        "html", "css", "pandas", "numpy"
    ]
    skills = []
    for s in skill_names:
        if s in t:
            display = {"sklearn": "Scikit-learn", "node": "Node.js",
                       "artificial intelligence": "Artificial Intelligence"}.get(s, s)
            if display not in skills:
                skills.append(display)
    skills_text = ", ".join(skills) if skills else "Python, Machine Learning"

    if any(x in t for x in ["b.tech", "btech", "bachelor of technology"]):
        education = "B.Tech"
    elif any(x in t for x in ["m.tech", "mtech", "master of technology"]):
        education = "M.Tech"
    elif "mba" in t:
        education = "MBA"
    elif "phd" in t or "doctorate" in t:
        education = "PhD"
    else:
        education = "B.Sc"

    certification = "No Certification"
    if any(x in t for x in ["certified", "certification", "certificate"]):
        certification = "Certification Found"

    roles = [
        ("machine learning engineer", "Machine Learning Engineer"),
        ("artificial intelligence engineer", "AI Engineer"),
        ("ai engineer", "AI Engineer"),
        ("data scientist", "Data Scientist"),
        ("data analyst", "Data Analyst"),
        ("full stack developer", "Full Stack Developer"),
        ("frontend developer", "Frontend Developer"),
        ("backend developer", "Backend Developer"),
        ("python developer", "Python Developer"),
        ("devops engineer", "DevOps Engineer"),
        ("data engineer", "Data Engineer"),
        ("software developer", "Software Developer"),
        ("software engineer", "Software Engineer"),
        ("web developer", "Web Developer"),
    ]
    job_role = "Software Engineer"
    for keyword, role in roles:
        if keyword in t:
            job_role = role
            break

    projects = 3
    m = re.search(r"(\d+)\s*\+?\s*projects?", t)
    if m:
        projects = int(m.group(1))
    projects = max(0, min(projects, 100))

    return {
        "experience": experience,
        "skills": skills_text,
        "education": education,
        "certification": certification,
        "job_role": job_role,
        "projects": projects,
    }


def calculate_candidate_score(experience, projects, skills, education, certification):
    score = min(experience * 5, 25)
    score += min(projects * 4, 20)
    skill_count = len([x for x in skills.split(",") if x.strip()])
    score += min(skill_count * 5, 25)
    score += {"B.Sc": 10, "B.Tech": 15, "M.Tech": 18, "MBA": 15, "PhD": 20}.get(education, 10)
    if certification != "No Certification":
        score += 10
    return min(score, 100)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🤖 AI Hiring System")
    st.divider()
    st.subheader("🧠 Model Information")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Test Accuracy:** 99.50%")
    st.write("**Training Samples:** 800")
    st.write("**Testing Samples:** 200")
    st.divider()
    st.subheader("📊 Prediction Features")
    for item in ["Experience", "Salary Expectation", "Projects", "AI Score", "Skills", "Education", "Certifications", "Job Role"]:
        st.write(f"• {item}")

# ---------------- TITLE + DASHBOARD ----------------
st.title("🤖 AI-Based Hiring Prediction System")
st.write("An end-to-end Machine Learning system for predicting whether a candidate should be Hired or Rejected.")
st.divider()

history_df = pd.DataFrame(st.session_state.prediction_history)
if history_df.empty:
    total = hired = rejected = 0
    rate = avg_score = 0
else:
    total = len(history_df)
    hired = history_df["Decision"].eq("Hire").sum()
    rejected = history_df["Decision"].eq("Reject").sum()
    rate = hired / total * 100
    avg_score = history_df["Resume Score"].mean()

st.subheader("📊 Hiring Dashboard")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("👥 Total Candidates", total)
c2.metric("✅ Hired", hired)
c3.metric("❌ Rejected", rejected)
c4.metric("📈 Hiring Rate", f"{rate:.2f}%")
c5.metric("⭐ Avg Resume Score", f"{avg_score:.1f}/100")
st.divider()

# ---------------- RESUME UPLOAD ----------------
st.header("📄 Resume Upload")
uploaded = st.file_uploader("Upload Candidate Resume (PDF)", type=["pdf"], key="resume_uploader")

if uploaded is not None:
    resume_id = (uploaded.name, uploaded.size)
    if st.session_state.resume_processed != resume_id:
        try:
            text = ""
            with pdfplumber.open(uploaded) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            st.session_state.resume_text = text
            st.session_state.resume_processed = resume_id
            st.session_state.resume_info = extract_resume_info(text) if text.strip() else None
        except Exception as error:
            st.error("❌ Could not read the uploaded PDF.")
            st.exception(error)
            st.stop()

    if st.session_state.resume_text.strip() and st.session_state.resume_info:
        st.success(f"✅ Resume uploaded: {uploaded.name}")
        with st.expander("📄 View Extracted Resume Text"):
            st.text_area("Resume Content", st.session_state.resume_text, height=250, disabled=True)

        info = st.session_state.resume_info
        score = calculate_candidate_score(info["experience"], info["projects"], info["skills"], info["education"], info["certification"])
        st.subheader("⭐ Candidate Score")
        sc1, sc2 = st.columns(2)
        sc1.metric("Resume-Based Score", f"{score}/100")
        if score >= 80:
            sc2.success("🏆 Strong Candidate")
        elif score >= 60:
            sc2.warning("🟡 Average Candidate")
        else:
            sc2.error("🔴 Needs Improvement")
        st.progress(score)

        st.subheader("🤖 AI Extracted Information")
        ec1, ec2 = st.columns(2)
        ec1.write(f"**Experience:** {info['experience']} years")
        ec1.write(f"**Skills:** {info['skills']}")
        ec1.write(f"**Education:** {info['education']}")
        ec2.write(f"**Certification:** {info['certification']}")
        ec2.write(f"**Job Role:** {info['job_role']}")
        ec2.write(f"**Projects:** {info['projects']}")

        if st.button("✨ Apply Extracted Resume Data", use_container_width=True):
            st.session_state.experience = info["experience"]
            st.session_state.projects = info["projects"]
            st.session_state.skills = info["skills"]
            st.session_state.education = info["education"]
            st.session_state.certification = info["certification"]
            st.session_state.job_role = info["job_role"]
            st.rerun()
    else:
        st.warning("⚠️ Resume uploaded, but no readable text was found. Try a text-based PDF.")

# ---------------- CANDIDATE FORM ----------------
st.header("📋 Candidate Information")
f1, f2 = st.columns(2)

with f1:
    experience = st.number_input("Experience (Years)", 0, 50, key="experience", step=1)
    salary = st.number_input("Salary Expectation ($)", 0, key="salary", value=50000, step=1000)
    projects = st.number_input("Projects Count", 0, 100, key="projects", step=1)
    ai_score = st.slider("AI Score (0-100)", 0, 100, key="ai_score", value=70)

with f2:
    skills = st.text_input("Skills", key="skills")
    education = st.selectbox("Education", ["B.Sc", "B.Tech", "M.Tech", "MBA", "PhD"], key="education")
    certification = st.text_input("Certification", key="certification")
    job_role = st.text_input("Job Role", key="job_role")

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Hiring Decision", use_container_width=True, type="primary"):
    candidate = pd.DataFrame({
        "Skills": [skills],
        "Experience (Years)": [experience],
        "Education": [education],
        "Certifications": [certification],
        "Job Role": [job_role],
        "Salary Expectation ($)": [salary],
        "Projects Count": [projects],
        "AI Score (0-100)": [ai_score],
    })

    categorical = ["Skills", "Education", "Certifications", "Job Role"]
    numerical = ["Experience (Years)", "Salary Expectation ($)", "Projects Count", "AI Score (0-100)"]

    try:
        encoded = encoder.transform(candidate[categorical])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical))
        final_features = pd.concat([candidate[numerical].reset_index(drop=True), encoded_df], axis=1)
        if hasattr(model, "feature_names_in_"):
            final_features = final_features.reindex(columns=list(model.feature_names_in_), fill_value=0)
        prediction = model.predict(final_features)
        probabilities = model.predict_proba(final_features)
    except Exception as error:
        st.error("❌ Prediction failed. This is usually caused by a value not matching the trained encoder.")
        st.exception(error)
        st.stop()

    classes = list(model.classes_)
    hire_probability = probabilities[0][classes.index(1)] * 100 if 1 in classes else 0
    reject_probability = probabilities[0][classes.index(0)] * 100 if 0 in classes else 0
    decision = "Hire" if prediction[0] == 1 else "Reject"

    resume_score = calculate_candidate_score(experience, projects, skills, education, certification)
    record = {
        "Experience": experience, "Salary": salary, "Projects": projects, "AI Score": ai_score,
        "Resume Score": resume_score, "Skills": skills, "Education": education,
        "Certification": certification, "Job Role": job_role, "Decision": decision,
        "Hire Probability": round(hire_probability, 2), "Reject Probability": round(reject_probability, 2)
    }
    st.session_state.prediction_history.append(record)

    st.divider()
    st.header("🎯 Prediction Result")
    if decision == "Hire":
        st.success("🎉 CANDIDATE RECOMMENDED FOR HIRING")
    else:
        st.error("❌ CANDIDATE NOT RECOMMENDED FOR HIRING")

    r1, r2 = st.columns(2)
    r1.metric("✅ Hire Probability", f"{hire_probability:.2f}%")
    r2.metric("❌ Reject Probability", f"{reject_probability:.2f}%")
    st.subheader("📊 Hiring Probability")
    st.progress(min(max(int(hire_probability), 0), 100))

    summary = pd.DataFrame({
        "Feature": ["Experience", "Salary Expectation", "Projects", "AI Score", "Resume Score", "Skills", "Education", "Certification", "Job Role", "Final Decision"],
        "Value": [f"{experience} years", f"${salary}", projects, f"{ai_score}/100", f"{resume_score}/100", skills, education, certification, job_role, decision]
    })
    st.subheader("👤 Candidate Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    result = pd.DataFrame([record])
    st.subheader("📥 Export Prediction")
    st.download_button("📥 Download Prediction Report", result.to_csv(index=False), "candidate_prediction.csv", "text/csv", use_container_width=True)

# ---------------- HISTORY ----------------
st.divider()
st.header("📚 Prediction History")
history_df = pd.DataFrame(st.session_state.prediction_history)

if not history_df.empty:
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    st.download_button("📥 Download Prediction History", history_df.to_csv(index=False), "prediction_history.csv", "text/csv", use_container_width=True)

    st.subheader("📈 Hiring Statistics")
    total = len(history_df)
    hired = history_df["Decision"].eq("Hire").sum()
    rejected = history_df["Decision"].eq("Reject").sum()
    rate = hired / total * 100 if total else 0
    avg_score = history_df["Resume Score"].mean()
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total", total)
    s2.metric("Hired", hired)
    s3.metric("Rejected", rejected)
    s4.metric("Hiring Rate", f"{rate:.2f}%")
    s5.metric("Avg Resume Score", f"{avg_score:.1f}/100")

    st.subheader("📊 Hiring Decision Distribution")
    chart = pd.DataFrame({"Decision": ["Hired", "Rejected"], "Candidates": [hired, rejected]})
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(chart["Decision"], chart["Candidates"])
    ax.set_xlabel("Hiring Decision")
    ax.set_ylabel("Number of Candidates")
    ax.set_title("Hiring Decision Distribution")
    st.pyplot(fig)
    plt.close(fig)

    if st.button("🗑️ Clear Prediction History", use_container_width=True):
        st.session_state.prediction_history = []
        st.rerun()
else:
    st.info("No predictions yet. Make a prediction to see your history here.")

st.divider()
st.caption("🤖 AI-Based Hiring Prediction System | Machine Learning Project | Logistic Regression") 
