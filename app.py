import streamlit as st
import joblib
import numpy as np

# Load the best model
model = joblib.load('GradientBoostingRegressor_model.pkl')
scaler = joblib.load('scaler.pkl')

sex_map ={'Female': 0, 'Male': 1}
smoker_map = {'No': 0, 'Yes': 1}
regions = ['NorthEast', 'NorthWest', 'SouthEast', 'SouthWest']

image_url = "https://miro.medium.com/v2/0*ZfFdinuIaBOVxWwp.jpg"

def prepare_input(age, bmi, children, sex, smoker, region):
    region_NorthWest = 1 if region == 'NorthWest' else 0
    region_SouthEast = 1 if region == 'SouthEast' else 0
    region_SouthWest = 1 if region == 'SouthWest' else 0

    input_data = np.array([[np.float64(age), bmi, np.log1p(np.float64(children)), sex, smoker, region_NorthWest, region_SouthEast, region_SouthWest]])
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled

def main():
    st.title("Insurance Charges Prediction", text_alignment="center")
    st.markdown(
        f'<img src="{image_url}" style="width:800px; height:350px; object-fit:cover;">',
        unsafe_allow_html=True
    )

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider('Age', min_value=15, max_value=65, value=30, step=1)
            sex = st.radio("Sex", options=list(sex_map.keys()))
            bmi = st.number_input('BMI', min_value=15.0, max_value=50.0, value=30.0, step=0.1)
        with col2:
            children = st.selectbox('Dependent Children', options=[0, 1, 2, 3, 4, 5], index=0)
            smoker = st.radio("Smoker", options=list(smoker_map.keys()))
            region = st.selectbox('Region', options=regions, index=0)            
        lcol1, lcol2, lcol3 = st.columns([2, 1, 2])
        with lcol2:
            submitted = st.form_submit_button("Predict")
    if submitted:
        input_data_scaled = prepare_input(age, bmi, children, sex_map[sex], smoker_map[smoker], region)
        predicted_charges = model.predict(input_data_scaled)[0]
        print(f"Predicted Insurance Charges: ${predicted_charges:,.2f}")
        st.success(f"Predicted Insurance Charges: ${predicted_charges:,.2f}")

if __name__ == "__main__":
    main()
