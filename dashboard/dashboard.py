import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
base_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "model", "student_model.pkl")
model = joblib.load(model_path)

# ---------- HEADER ----------
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>
    🎓 Student Performance Prediction System
    </h1>
    <p style='text-align: center; font-size:18px;'>
    Predict whether a student will Pass or be At Risk using ML
    </p>
""", unsafe_allow_html=True)

st.divider()

# ---------- LAYOUT ----------
col1, col2 = st.columns(2)

# ---------- INPUT SECTION ----------
with col1:

    st.markdown("""
    <div style="
        background-color:#262730;
        padding:20px;
        border-radius:15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
    ">
    <h3 style="color:#4CAF50; text-align:center;">📊 Enter Student Details</h3>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2 = st.columns(2)

    with c1:
        gpa = st.slider("🎓 Prior GPA", 0.0, 4.0, 2.5)
        attendance = st.slider("📅 Attendance %", 0, 100, 70)
        quiz = st.slider("📝 Quiz Average", 0, 100, 60)
        assign = st.slider("📚 Assignment Avg", 0, 100, 65)
        midterm = st.slider("📊 Midterm Marks", 0, 100, 60)

    with c2:
        study = st.slider("⏱ Study Hours/week", 0, 20, 8)
        lms = st.slider("💻 LMS Logins/week", 0, 10, 3)
        forum = st.slider("💬 Forum Posts", 0, 10, 1)
        commute = st.slider("🚗 Commute Time (min)", 0, 120, 30)

    st.write("")

    predict_btn = st.button("🚀 Predict Performance")

# ---------- RESULT SECTION ----------
with col2:

    st.markdown("""
    <div style="
        background-color:#262730;
        padding:20px;
        border-radius:15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
    ">
    <h3 style="color:#4CAF50; text-align:center;">📈 Prediction Result</h3>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if predict_btn:

        data = [[gpa, attendance, quiz, assign, midterm, study, lms, forum, commute]]

        result = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        if result == 1:
            st.success(f"✅ Student will PASS\n\nConfidence: {prob*100:.2f}%")
        else:
            st.error(f"⚠️ Student is AT RISK\n\nConfidence: {(1-prob)*100:.2f}%")

        # ---------- GRAPH ----------
        st.subheader("📊 Performance Overview")

        features = [
            "GPA","Attendance","Quiz","Assignment","Midterm",
            "Study Hours","LMS","Forum","Commute"
        ]

        values = [gpa, attendance, quiz, assign, midterm, study, lms, forum, commute]

        fig, ax = plt.subplots()
        ax.barh(features, values)
        ax.set_title("Student Input Features")

        st.pyplot(fig)

# ---------- FOOTER ----------
st.divider()
st.markdown("""
<p style='text-align: center;'>
Made with ❤️ by Akeel | Machine Learning Project
</p>
""", unsafe_allow_html=True)