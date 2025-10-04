import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.app_logo import add_logo

MODEL_PATH = r"D:\Major Project\Mental-Health-Prediction-ML\quantum (1).joblib "
model = joblib.load(MODEL_PATH)

def handle_gender(gender):
    return 1 if gender.lower() == 'male' else 0

def handle_no_employees(no_employees):
    mapping = {'1-5': 0, '6-25': 1, '26-100': 2, '500-1000': 3, 'More than 1000': 4}
    return mapping.get(no_employees, 0)

def handle_mental_health_consequence(value):
    return {'No': 0, 'Yes': 1}.get(value, 2)

mappings = {
    'self_employed': {'No': 0, 'Yes': 1},
    'family_history': {'No': 0, 'Yes': 1},
    'work_interfere': {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3},
    'tech_company': {'No': 0, 'Yes': 1},
    'benefits': {'No': 0, "Don't know": 1, 'Yes': 2},
    'remote_work': {'No': 0, 'Yes': 1},
}

st.set_page_config(page_title="Mental Health Prediction", page_icon="🧠", layout="wide")
st.title("Mental Health Prediction Model")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    self_employed = st.selectbox("Self Employed", ['No', 'Yes'])
    family_history = st.selectbox("Family History", ['No', 'Yes'])
    work_interfere = st.selectbox("Work Interference", ['Never', 'Rarely', 'Sometimes', 'Often'])
    no_employees = st.selectbox("Number of Employees", ['1-5', '6-25', '26-100', '500-1000', 'More than 1000'])

with col2:
    tech_company = st.selectbox("Tech Company", ['No', 'Yes'])
    benefits = st.selectbox("Benefits", ['No', "Don't know", 'Yes'])
    remote_work = st.selectbox("Remote Work", ['No', 'Yes'])
    mental_health_consequence = st.selectbox("Mental Health Consequence", ['No', 'Yes', 'Maybe'])
    phys_health_consequence = st.selectbox("Physical Health Consequence", ['No', 'Yes', 'Maybe'])
    heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=70)

st.markdown("---")

if st.button("🔍 Predict", use_container_width=True):
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [handle_gender(gender)],
        "self_employed": [mappings['self_employed'][self_employed]],
        "family_history": [mappings['family_history'][family_history]],
        "work_interfere": [mappings['work_interfere'][work_interfere]],
        "no_employees": [handle_no_employees(no_employees)],
        "tech_company": [mappings['tech_company'][tech_company]],
        "benefits": [mappings['benefits'][benefits]],
        "remote_work": [mappings['remote_work'][remote_work]],
        "mental_health_consequence": [handle_mental_health_consequence(mental_health_consequence)],
        "phys_health_consequence": [handle_mental_health_consequence(phys_health_consequence)],
        "Heart_Rate": [heart_rate]
    })

    input_arr = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_arr)
    result = "Likely to face Mental Health Issues" if prediction[0] == 1 else "Unlikely to face Mental Health Issues"
    
    if prediction[0] == 1:
        st.error(f"### Prediction: {result}")
    else:
        st.success(f"### Prediction: {result}")
    
    st.markdown("---")

    labels = ['Unlikely', 'Likely']
    values = [100 - prediction[0] * 100, prediction[0] * 100]
    colors = ['#4CAF50', '#FF5733']
    
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.4, marker=dict(colors=colors)))
    fig.update_layout(title_text="Mental Health Prediction Probability", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    heart_rate_data = np.random.normal(loc=70, scale=15, size=500)
    fig_hr = px.histogram(x=heart_rate_data, nbins=40, title="Heart Rate Distribution", color_discrete_sequence=['#1f77b4'])
    fig_hr.add_vline(x=heart_rate, line_dash="dash", line_color="red", annotation_text="Your Heart Rate")
    st.plotly_chart(fig_hr, use_container_width=True)
    
    st.markdown("### 📝 Recommendations")
    if prediction[0] == 1:
        st.warning("- **Seek Professional Help**: Consulting a therapist can be beneficial.")
        st.warning("- **Practice Mindfulness**: Meditation and breathing exercises can help.")
        st.warning("- **Maintain a Healthy Routine**: Exercise, sleep, and a balanced diet are essential.")
    else:
        st.info("- **Continue Healthy Practices**: Keep maintaining your good mental health.")
        st.info("- **Engage in Social Activities**: Connecting with others boosts well-being.")
        st.info("- **Monitor Stress Levels**: Avoid burnout and maintain a healthy work-life balance.")
