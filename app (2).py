
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder # Needed if you want to inspect or use LabelEncoder directly

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
label_encoders_from_bundle = bundle['label_encoders'] # Use pre-fitted encoders from bundle
original_columns = bundle['original_columns']

# Load the original dataset to get unique categorical values for Streamlit selectbox options
# This is loaded separately and not used to re-fit encoders.
try:
    original_df_for_options = pd.read_csv('/content/Salary_Dataset_DataScienceLovers.csv')
    # Ensure null values are handled for consistent option display
    for column in original_df_for_options.columns:
        if original_df_for_options[column].dtype == 'object':
            original_df_for_options[column] = original_df_for_options[column].fillna(original_df_for_options[column].mode()[0])
except FileNotFoundError:
    st.error("Error: Original dataset '/content/Salary_Dataset_DataScienceLovers.csv' not found. Please ensure it's in the correct path.")
    st.stop()


st.title('Salary Prediction for Data Science Roles')
st.write('Enter the details below to predict the salary.')

# Input fields
rating = st.slider('Rating', 1.0, 5.0, 3.9)

# For categorical features, use selectbox and then encode
company_name_options = sorted(original_df_for_options['Company Name'].astype(str).unique())
company_name_input = st.selectbox('Company Name', company_name_options)

job_title_options = sorted(original_df_for_options['Job Title'].astype(str).unique())
job_title_input = st.selectbox('Job Title', job_title_options)

salaries_reported = st.number_input('Salaries Reported', min_value=1, value=1)

location_options = sorted(original_df_for_options['Location'].astype(str).unique())
location_input = st.selectbox('Location', location_options)

employment_status_options = sorted(original_df_for_options['Employment Status'].astype(str).unique())
employment_status_input = st.selectbox('Employment Status', employment_status_options)

job_roles_options = sorted(original_df_for_options['Job Roles'].astype(str).unique())
job_roles_input = st.selectbox('Job Roles', job_roles_options)

# Create a DataFrame for the input
input_data_raw = pd.DataFrame([
    {
        'Rating': rating,
        'Company Name': company_name_input,
        'Job Title': job_title_input,
        'Salaries Reported': salaries_reported,
        'Location': location_input,
        'Employment Status': employment_status_input,
        'Job Roles': job_roles_input
    }
])

# Preprocess input data using the loaded encoders and scaler
def preprocess_input(df_input_raw, label_encoders, scaler_X, original_columns):
    df_processed = df_input_raw.copy()

    # Apply label encoding using the pre-fitted encoders
    for col, le in label_encoders.items():
        if col in df_processed.columns:
            try:
                # Transform if the category is known
                df_processed[col] = le.transform(df_processed[col])
            except ValueError:
                # Handle unseen categories by assigning -1
                df_processed[col] = -1

    # Ensure all original columns are present and in the correct order
    for col in original_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0 # Or a sensible default if feature was missing
    df_processed = df_processed[original_columns] # Reorder columns to match training

    # Scale features using the pre-fitted scaler
    df_scaled = scaler_X.transform(df_processed)
    return df_scaled

if st.button('Predict Salary'):
    try:
        # Pass the pre-fitted label_encoders_from_bundle
        processed_input = preprocess_input(input_data_raw, label_encoders_from_bundle, scaler_X, original_columns)
        prediction_scaled = model.predict(processed_input)
        predicted_salary = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
        st.success(f'Predicted Salary: ₹{predicted_salary:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please ensure all inputs are valid and the model bundle is correctly loaded.")
