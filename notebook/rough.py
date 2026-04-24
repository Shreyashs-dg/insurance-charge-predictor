import numpy as np  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score

# Load dataset
df=pd.read_csv("insurance.csv")

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
# print(df.isna().sum().sum())# checked no null value's 
# print(df.duplicated().sum().sum()) # no duplicatedd value's

# Convert categorical to numeric (label encoding)
le=LabelEncoder()

for i in df.columns:
    if df[i].dtype == 'object':
        df[i]= le.fit_transform(df[i])

#to check the coded value to orginal value  maping=dict(zip(le.classes_ ,le.transform(le.classes_)))
#Encodeing the data from object to numeric type is converted sucessfully

#vif part make each columns vif score is around 5

# now we will remove one by one 
df=df.drop(['num_of_steps'],axis=1)
df=df.drop(['Anual_Salary'],axis=1)
df=df.drop(['bmi'],axis=1)
df=df.drop(['NUmber_of_past_hospitalizations'],axis=1)
df=df.drop(['past_consultations'],axis=1)
#Now vfc score is less than 5-6 it is ok 

x=df.drop(['charges'],axis=1)
vif=pd.DataFrame()
vif['Column_name']=x.columns
vif['vfc_score']=[ variance_inflation_factor(x.values,i) for i in range(len(x.columns))]

#model buding 
X=df.drop('charges',axis=1)
Y=df['charges']

model= LinearRegression()

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=7)
model.fit(x_train,y_train)

prediction=model.predict(x_test)

#check the r2 score 
print(r2_score(y_test,prediction4))#score is 0.854 so it is a good model

# Features & target
X = df.drop("charges", axis=1)
y = df["charges"]


# Sample prediction
sample = X_test.iloc[0].values.reshape(1, -1)
prediction = model.predict(sample)

print("Predicted Charge:", prediction[0])
Claim_Amount,Hospital_expenditure ,region

def user_input():
    age = int(input("Enter age: "))
    sex = input("Male/Female: ").lower()
    bmi = float(input("Enter BMI: "))
    children = int(input("Enter number of children: "))
    smoker = input("Smoker (yes/no): ").lower()

    user_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    return user_df

user_data = user_input()
prediction = model.predict(user_data)

print("Estimated Insurance Charge:", prediction[0])