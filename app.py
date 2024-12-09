import streamlit as st
import pickle
import numpy as np

# Load the trained model (replace 'model.pkl' with your actual file name)
with open('hhmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the customized ranges for each feature based on dataset statistics
custom_ranges = {
    'Engine rpm': (61.0, 2239.0),
    'Lub oil pressure': (0.003384, 7.265566),
    'Fuel pressure': (0.003187, 21.138326),
    'Coolant pressure': (0.002483, 7.478505),
    'lub oil temp': (71.321974, 89.580796),
    'Coolant temp': (61.673325, 195.527912),
    'Temperature_difference': (-22.669427, 119.008526)
}

# Feature Descriptions
feature_descriptions = {
    'Engine rpm': 'Revolution per minute of the engine.',
    'Lub oil pressure': 'Pressure of the lubricating oil.',
    'Fuel pressure': 'Pressure of the fuel.',
    'Coolant pressure': 'Pressure of the coolant.',
    'lub oil temp': 'Temperature of the lubricating oil.',
    'Coolant temp': 'Temperature of the coolant.',
    'Temperature_difference': 'Temperature difference between components.'
}

# Engine Condition Prediction App
def main():
    st.title("Engine Condition Prediction")

    # Display feature descriptions
    st.sidebar.title("Feature Descriptions")
    for feature, description in feature_descriptions.items():
        st.sidebar.markdown(f"**{feature}:** {description}")

    # Input widgets with customized ranges
    if 'engine_rpm' not in st.session_state:
        st.session_state['engine_rpm'] = (custom_ranges['Engine rpm'][0] + custom_ranges['Engine rpm'][1]) / 2
    if 'lub_oil_pressure' not in st.session_state:
        st.session_state['lub_oil_pressure'] = (custom_ranges['Lub oil pressure'][0] + custom_ranges['Lub oil pressure'][1]) / 2
    if 'fuel_pressure' not in st.session_state:
        st.session_state['fuel_pressure'] = (custom_ranges['Fuel pressure'][0] + custom_ranges['Fuel pressure'][1]) / 2
    if 'coolant_pressure' not in st.session_state:
        st.session_state['coolant_pressure'] = (custom_ranges['Coolant pressure'][0] + custom_ranges['Coolant pressure'][1]) / 2
    if 'lub_oil_temp' not in st.session_state:
        st.session_state['lub_oil_temp'] = (custom_ranges['lub oil temp'][0] + custom_ranges['lub oil temp'][1]) / 2
    if 'coolant_temp' not in st.session_state:
        st.session_state['coolant_temp'] = (custom_ranges['Coolant temp'][0] + custom_ranges['Coolant temp'][1]) / 2
    if 'temp_difference' not in st.session_state:
        st.session_state['temp_difference'] = (custom_ranges['Temperature_difference'][0] + custom_ranges['Temperature_difference'][1]) / 2

    # Sliders to adjust values
    engine_rpm = st.slider("Engine RPM", min_value=float(custom_ranges['Engine rpm'][0]), 
                           max_value=float(custom_ranges['Engine rpm'][1]), 
                           value=st.session_state['engine_rpm'])
    lub_oil_pressure = st.slider("Lub Oil Pressure", min_value=custom_ranges['Lub oil pressure'][0], 
                                 max_value=custom_ranges['Lub oil pressure'][1], 
                                 value=st.session_state['lub_oil_pressure'])
    fuel_pressure = st.slider("Fuel Pressure", min_value=custom_ranges['Fuel pressure'][0], 
                              max_value=custom_ranges['Fuel pressure'][1], 
                              value=st.session_state['fuel_pressure'])
    coolant_pressure = st.slider("Coolant Pressure", min_value=custom_ranges['Coolant pressure'][0], 
                                 max_value=custom_ranges['Coolant pressure'][1], 
                                 value=st.session_state['coolant_pressure'])
    lub_oil_temp = st.slider("Lub Oil Temperature", min_value=custom_ranges['lub oil temp'][0], 
                             max_value=custom_ranges['lub oil temp'][1], 
                             value=st.session_state['lub_oil_temp'])
    coolant_temp = st.slider("Coolant Temperature", min_value=custom_ranges['Coolant temp'][0], 
                             max_value=custom_ranges['Coolant temp'][1], 
                             value=st.session_state['coolant_temp'])
    temp_difference = st.slider("Temperature Difference", min_value=custom_ranges['Temperature_difference'][0], 
                                max_value=custom_ranges['Temperature_difference'][1], 
                                value=st.session_state['temp_difference'])

    # Update session state with new values
    st.session_state['engine_rpm'] = engine_rpm
    st.session_state['lub_oil_pressure'] = lub_oil_pressure
    st.session_state['fuel_pressure'] = fuel_pressure
    st.session_state['coolant_pressure'] = coolant_pressure
    st.session_state['lub_oil_temp'] = lub_oil_temp
    st.session_state['coolant_temp'] = coolant_temp
    st.session_state['temp_difference'] = temp_difference

    # Predict button
    if st.button("Predict Engine Condition"):
        result, confidence = predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference)

        # Explanation
        if result == 0:
            st.info(f"The engine is predicted to be in a normal condition. The Confidence level is: {1.0 - confidence:.2%}")
        else:
            st.warning(f"Warning! Please investigate further. The Confidence level is: {1.0 - confidence:.2%}")

    # Reset button
    if st.button("Reset Values"):
        # Reset the session state to default values
        for feature in custom_ranges:
            st.session_state[feature] = (custom_ranges[feature][0] + custom_ranges[feature][1]) / 2
        st.experimental_rerun()  # Rerun to refresh the values

# Function to predict engine condition
def predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference):
    input_data = np.array([engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp, temp_difference]).reshape(1, -1)
    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data)[:, 1]  # For binary classification, adjust as needed
    return prediction[0], confidence[0]

if __name__ == "__main__":
    main()
