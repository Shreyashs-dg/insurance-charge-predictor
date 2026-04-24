import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load and Train Model
df = pd.read_csv("../data/insurance.csv")
df.columns = df.columns.str.strip()

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

X = df[["age","sex","bmi","children","smoker","region"]]
y = np.log(df["charges"])

ct = ColumnTransformer(
    [("encoder", OneHotEncoder(drop="first"), ["sex","smoker","region"])],
    remainder="passthrough"
)

X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# UI Starts Here
st.title("Insurance Charge Prediction App")

st.write("Enter details below to predict insurance charges")

# Inputs
age = st.slider("Age", 18, 65, 25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 15.0, 40.0, 25.0)
children = st.slider("Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northwest","northeast","southeast","southwest"])

# Predict Button
if st.button("Predict Charges"):
    user_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    user_transformed = ct.transform(user_df)
    prediction = model.predict(user_transformed)

    final = np.exp(prediction[0])

    st.success(f"💰 Estimated Charges: ₹ {round(final, 2)}")