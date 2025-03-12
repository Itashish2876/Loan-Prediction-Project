import streamlit as st
import numpy as np
import pickle




# Custom CSS for better layout and colors
st.markdown(
    '''
    <style>
    .stButton>button {
            background-color: #007BFF;
            color: white;
            padding: 8px 16px;
            border-radius: 5px;
        }
    </style>
    ''', unsafe_allow_html=True
)
st.markdown('<div class="main">', unsafe_allow_html=True)





# ===========================
# ✅ Add a title and description  # Start of the App
# ===========================

st.title("🏦 Loan Eligibility Prediction App")
st.write("""
💰 This AI-powered **Loan Eligibility Prediction Model** helps you determine your **eligibility for a loan** based on key financial parameters such as 
**income, credit score, loan amount** .
📊 Simply enter your details, and the model will predict whether you are **eligible** for the loan or not!
""")
st.markdown("---")  







# ===========================
# 📜 Load Model and Scaler with Error Handling
# ===========================

try:
    model = pickle.load(open("Loan_Model.pkl", 'rb'))
    scaler = pickle.load(open("loan_scale.pkl", 'rb'))
except FileNotFoundError:
    st.error("Model or scaler file not found. Please check the files.")
    st.stop()




 
# ===========================
# 📊 Input Section  (with basic validation ranges provided in sidebar)
# ===========================

st.sidebar.title("ℹ️ Input Guidelines")
st.sidebar.write("**Age:** Above 18 years")
st.sidebar.write("**Income:** Above 25000 ")
st.sidebar.write("**Credit Score:** Above 560")



# Input Fields (Two Columns for better layout)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Enter the Age:", min_value=18, max_value=100, value=30, step=1)
    income = st.number_input("Enter the Income (in USD):", min_value=25000.0, value=50000.0, step=1000.0)

with col2:
    gender = st.selectbox("Select Gender:", options=['Male', 'Female'])
    credit_score = st.number_input("Enter the Credit Score:", min_value=560, max_value=900, value=650, step=1)



    
# ========================================
# ✅ Helper Function for Prediction
# ========================================


# Encoding Gender (since model expects 0/1)
def encode_gender(gender):
    return 1 if gender == 'Male' else 0

# Prediction Function
def predict_loan_approval(age, gender, income, credit_score):
    gender_encoded = encode_gender(gender)

    # Prepare the feature array (matching model input structure)
    features = np.array([[age, gender_encoded, income, credit_score]])

    # Scale the features using the pre-fitted scaler
    scaled_features = scaler.transform(features)

    # Predict using the loaded model
    prediction = model.predict(scaled_features)

    return prediction[0]



# ========================================
# ✅ Predict Button with results block
# ========================================

if st.button('🔎 Predict Loan Eligibility'):
    result = predict_loan_approval(age, gender, income, credit_score)

    
    st.markdown("---")    # <- This adds a nice separator line
    
    if result == 1:
        st.error("❌ Loan Denied. Unfortunately, you are **not eligible** for the loan.")
    else:
        st.success("✅ Loan Approved! Congratulations, you are **eligible** for the loan.")

    st.markdown("Prediction is based on factors like age, income ,credit_score.")
    
# End of the App
st.markdown("</div>", unsafe_allow_html=True)