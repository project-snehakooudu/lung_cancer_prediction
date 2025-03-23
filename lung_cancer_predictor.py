import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import streamlit as st

# Custom styling for the Streamlit app
st.markdown(
    """
    <style>
        /* Set background color for the whole page */
        .stApp {
            background-color: white !important;
            color: black !important;
        }

        /* Ensure all text appears black */
        h1, h2, h3, h4, h5, h6, p, div, span {
            color: black !important;
        }

        /* Customize the button */
        .stButton>button {
            background-color: green !important;
            color: white !important;
            border-radius: 10px;
            border: 2px solid white;
            padding: 10px;
            font-size: 16px;
        }

        /* Button hover effect */
        .stButton>button:hover {
            background-color: darkgreen !important;
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def get_feature_importance(model, X):
    """Extract feature importance from the model"""
    feature_importance = model.named_steps['model'].feature_importances_
    feature_names = X.columns
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    
    return importance_df

def create_prediction_interface():
    """Create a Streamlit interface for users to input data and get predictions"""
    st.title('Lung Cancer Prediction Tool')
    st.write('Enter patient information to predict lung cancer risk')
    
    # Load the trained model
    model = joblib.load('lung_cancer_model.pkl')
    
    # Create input fields for each feature
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider('Age', 20, 90, 50)
        smoking = st.slider('Smoking (pack-years)', 0, 60, 0)
        yellow_fingers = st.checkbox('Yellow Fingers')
        anxiety = st.checkbox('Anxiety')
        peer_pressure = st.checkbox('Peer Pressure')
        chronic_disease = st.checkbox('Chronic Disease')
        fatigue = st.checkbox('Fatigue')
    
    with col2:
        allergy = st.checkbox('Allergy')
        wheezing = st.checkbox('Wheezing')
        alcohol = st.checkbox('Alcohol Consumption')
        coughing = st.checkbox('Coughing')
        shortness_of_breath = st.checkbox('Shortness of Breath')
        swallowing_difficulty = st.checkbox('Swallowing Difficulty')
        chest_pain = st.checkbox('Chest Pain')
    
    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        'Age': [age],
        'Smoking': [smoking],
        'YellowFingers': [int(yellow_fingers)],
        'Anxiety': [int(anxiety)],
        'PeerPressure': [int(peer_pressure)],
        'ChronicDisease': [int(chronic_disease)],
        'Fatigue': [int(fatigue)],
        'Allergy': [int(allergy)],
        'Wheezing': [int(wheezing)],
        'AlcoholConsuming': [int(alcohol)],
        'Coughing': [int(coughing)],
        'ShortnessOfBreath': [int(shortness_of_breath)],
        'SwallowingDifficulty': [int(swallowing_difficulty)],
        'ChestPain': [int(chest_pain)]
    })
    
    # Make prediction when user clicks the button
    if st.button('Predict'):
        # Get prediction and probability
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Display result
        st.subheader('Prediction Result')
        if prediction == 1:
            st.error(f'High risk of lung cancer detected (Probability: {probability:.2%})')
            st.write('Please consult with a healthcare professional for further evaluation.')
        else:
            st.success(f'Low risk of lung cancer (Probability: {probability:.2%})')
            st.write('Regular health check-ups are still recommended.')
        
        # Display risk factors
        st.subheader('Key Risk Factors:')
        importance_df = get_feature_importance(model, input_data)
        top_factors = importance_df.head(5)['Feature'].tolist()
        
        active_risk_factors = []
        for factor in top_factors:
            if factor == 'Age' and age > 50:
                active_risk_factors.append(f"Age ({age} years)")
            elif factor == 'Smoking' and smoking > 0:
                active_risk_factors.append(f"Smoking ({smoking} pack-years)")
            elif factor in input_data.columns and input_data[factor].values[0] == 1:
                active_risk_factors.append(factor)
        
        if active_risk_factors:
            st.write("Your highest risk factors:")
            for factor in active_risk_factors:
                st.write(f"- {factor}")
        else:
            st.write("No significant risk factors identified in your inputs.")

if __name__ == "__main__":
    create_prediction_interface()
