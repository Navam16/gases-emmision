
import streamlit as st
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("co2_model.joblib")
le_state = joblib.load("le_state.joblib")
le_industry = joblib.load("le_industry.joblib")
le_unit = joblib.load("le_unit.joblib")

# Custom CSS for background and styling
st.markdown(
    """
    <style>
    body {
        background-image: url("https://images.unsplash.com/photo-1528821154947-1aa3a65f744c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80");
        background-size: cover;
    }
    .main-container {
        background-color: rgba(255, 255, 255, 0.90);
        padding: 2rem;
        border-radius: 15px;
        max-width: 800px;
        margin: auto;
        box-shadow: 0px 0px 20px rgba(0,0,0,0.2);
    }
    h1 {
        color: #2E8B57;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.2em;
    }
    h4.tagline {
        color: #226644;
        font-style: italic;
        text-align: center;
        margin-top: 0;
    }
    footer {
        text-align: center;
        margin-top: 50px;
        color: #888888;
        font-size: 0.9em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# UI Section
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown("### 🌿 CO₂ Emission Predictor")
st.markdown('<h4 class="tagline">Empowering Greener Decisions Through Data 🌍</h4>', unsafe_allow_html=True)

# Dropdown values
states = le_state.classes_
industries = le_industry.classes_
unit_types = le_unit.classes_

state = st.selectbox("📍 Select State", states)
industry = st.selectbox("🏭 Select Industry Type", industries)
unit_type = st.selectbox("⚙️ Select Unit Type", unit_types)

heat = st.slider("🔥 Heat Input (mmBTU/hr)", min_value=0.0, max_value=500.0, value=285.0, step=1.0)
ch4 = st.slider("🧪 Methane (CH₄) Emissions", min_value=0.0, max_value=10.0, value=3.4, step=0.1)
n2o = st.slider("🌫️ Nitrous Oxide (N₂O) Emissions", min_value=0.0, max_value=10.0, value=4.4, step=0.1)

if st.button("🚀 Predict CO₂ Emissions"):
    s = le_state.transform([state])[0]
    i = le_industry.transform([industry])[0]
    u = le_unit.transform([unit_type])[0]

    input_data = np.array([[s, i, heat, u, ch4, n2o]])
    prediction = model.predict(input_data)

    st.success(f"🌿 Predicted CO₂ Emission: {prediction[0]:.2f} units")

st.markdown('<footer>Made by Navam | For a Greener Future 🌱</footer>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
