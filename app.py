import streamlit as st
import joblib
import numpy as np

# Load the best model
model = joblib.load('GradientBoostingRegressor_model.pkl')
scaler = joblib.load('scaler.pkl')

sex_map ={'Female': 0, 'Male': 1}
smoker_map = {'No': 0, 'Yes': 1}
region_map = {'NorthEast': 0, 'NorthWest': 1, 'SouthEast': 2, 'SouthWest': 3}

image_url = "https://miro.medium.com/v2/0*ZfFdinuIaBOVxWwp.jpg"


def main():
    st.title("Income Prediction", text_alignment="center")
    st.markdown(
        f'<img src="{image_url}" style="width:800px; height:350px; object-fit:cover;">',
        unsafe_allow_html=True
    )

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider('Age', min_value=15.0, max_value=65.0, value=30.0, step=1.0)
            sex = st.radio("Sex", options=list(sex_map.keys()))
            bmi = st.number_input('BMI', min_value=15.0, max_value=50.0, value=30.0, step=0.1)
        with col2:
            children = st.selectbox('Dependent Children', options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=0)
            smoker = st.radio("Smoker", options=list(smoker_map.keys()))
            region = st.selectbox('Region', options=list(region_map.keys()))            
        lcol1, lcol2, lcol3 = st.columns([2, 1, 2])
        with lcol2:
            submitted = st.form_submit_button("Predict")
    if submitted:
        input_data = np.array([[age, sex_map[sex], bmi, np.log1p(children), smoker_map[smoker], region_map[region]]])
        input_data_scaled = scaler.transform(input_data)
        predicted_charges = model.predict(input_data_scaled)[0]
        print(f"Predicted Insurance Charges: ${predicted_charges:,.2f}")
        # 5. INVERSE TRANSFORM THE TARGET: Convert logged decimals back to real values
        predicted_charges = np.expm1(predicted_charges)
        st.success(f"Predicted Insurance Charges: ${predicted_charges:,.2f}")

if __name__ == "__main__":
    main()
