
import streamlit as st
import pandas as pd
import pickle

# Load the deployment bundle
@st.cache_resource
def load_model_bundle():
    with open('streamlit_model_bundle.pkl', 'rb') as file:
        bundle = pickle.load(file)
    return bundle

bundle = load_model_bundle()
model = bundle['model']
scaler_X = bundle['scaler_X']
scaler_y = bundle['scaler_y']
label_encoders = bundle['label_encoders']
original_columns = bundle['original_columns']

st.title('Salary Prediction for Data Science Roles')
st.write('Enter the details below to predict the salary.')

# Input fields
rating = st.slider('Rating', 1.0, 5.0, 3.9)
company_name_str = st.text_input('Company Name', 'Google')
job_title_str = st.text_input('Job Title', 'Data Scientist')
salaries_reported = st.number_input('Salaries Reported', min_value=1, value=1)
location_str = st.text_input('Location', 'Bangalore')
employment_status_str = st.selectbox('Employment Status', ['Full Time', 'Intern', 'Contractor', 'Part Time'])
job_roles_str = st.text_input('Job Roles', 'Data Scientist')

# Create a DataFrame for the input
input_data = pd.DataFrame([
    {
        'Rating': rating,
        'Company Name': company_name_str,
        'Job Title': job_title_str,
        'Salaries Reported': salaries_reported,
        'Location': location_str,
        'Employment Status': employment_status_str,
        'Job Roles': job_roles_str
    }
])

# Preprocess input data
def preprocess_input(df_input, label_encoders, scaler_X, original_columns):
    df_processed = df_input.copy()

    # Apply label encoding
    for col, le in label_encoders.items():
        if col in df_processed.columns:
            try:
                # Transform if the category is known
                df_processed[col] = le.transform(df_processed[col])
            except ValueError:
                # Handle unseen categories by assigning a default value or warning
                # For simplicity, assign -1 or mode of the training data
                df_processed[col] = -1 # Or a more sophisticated handling

    # Ensure all original columns are present and in the correct order
    for col in original_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0 # Or a sensible default for missing features
    df_processed = df_processed[original_columns]

    # Scale numerical features
    df_scaled = scaler_X.transform(df_processed)
    return df_scaled

if st.button('Predict Salary'):
    try:
        processed_input = preprocess_input(input_data, label_encoders, scaler_X, original_columns)
        prediction_scaled = model.predict(processed_input)
        predicted_salary = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
        st.success(f'Predicted Salary: ₹{predicted_salary:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please ensure all inputs are valid and the model bundle is correctly loaded.")
