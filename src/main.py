import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

# 1. Load Dataset
df = pd.read_csv("data\insurance.csv")

#We will dpop unwanted columns according to the vif score
df = df.drop(columns=[
    'num_of_steps','Anual_Salary',
    'NUmber_of_past_hospitalizations',
    'past_consultations','Claim_Amount'
], errors='ignore')

#Now vfc score is less than 5-6 it is ok 

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# 2. Features & Target
X = df.drop("charges", axis=1)
y = df["charges"]

# Categorical columns
cat_cols = ["sex", "smoker", "region"]


# 3. Encoding (Proper way)
ct = ColumnTransformer(
    transformers=[
        ("encoder", OneHotEncoder(drop="first"), cat_cols)
    ],
    remainder="passthrough"
)

X = ct.fit_transform(X)

# 4. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
print("R2 Score: ", r2_score(y_test, y_pred))

# 6. User Input
def get_user_input():
    age = int(input("Enter age: "))
    sex = input("Male/Female: ").lower()
    children = int(input("Enter number of children: "))
    smoker = input("Smoker (yes/no): ").lower()
    region = input("Region (northwest/northeast/southeast/southwest): ").lower()
    Hospital_expenditure= float(input("Enter the hostel expendature : "))
    bmi=float(input("Enter the bmi(Body mass index) : "))

    user_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "children": children,
        "smoker": smoker,
        "region": region,
        'Hospital_expenditure ':Hospital_expenditure,
        'bmi':bmi
    }])

    return user_df

# 7. Prediction
user_data = get_user_input()

# Apply transformation
user_transformed = ct.transform(user_data)
prediction = model.predict(user_transformed)

print("\nEstimated Insurance Charge: ₹", round(prediction[0], 2))
