
import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load('model_compressed.pkl')
le_state = joblib.load('le_state.pkl')
le_industry = joblib.load('le_industry.pkl')
le_unit = joblib.load('le_unit.pkl')

# Get dropdown options
states = le_state.classes_.tolist()
industries = le_industry.classes_.tolist()
unit_types = le_unit.classes_.tolist()

# Streamlit app layout
st.title("🌍 CO₂ Emission Predictor")
st.markdown("Select values from dropdowns and adjust sliders to predict emissions.")

state = st.selectbox("State", states)
industry = st.selectbox("Industry", industries)
unit_type = st.selectbox("Unit Type", unit_types)

heat = st.slider("Heat Input (mmBTU/hr)", min_value=0.0, max_value=500.0, value=285.0, step=1.0)
ch4 = st.slider("CH₄ Emissions", min_value=0.0, max_value=10.0, value=3.4, step=0.1)
n2o = st.slider("N₂O Emissions", min_value=0.0, max_value=10.0, value=4.4, step=0.1)

if st.button("Predict CO2"):
    try:
        s = le_state.transform([state])[0]
        i = le_industry.transform([industry])[0]
        u = le_unit.transform([unit_type])[0]
        
        input_data = np.array([[s, i, heat, u, ch4, n2o]])
        prediction = model.predict(input_data)[0]
        
        st.success(f"🌿 Predicted CO₂ Emission: **{prediction:.2f}** units")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
