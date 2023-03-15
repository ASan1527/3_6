import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

def data_deal(df):
    df.drop("Loan_ID", axis=1, inplace=True)
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    # 分箱处理
    df1 = df.loc[:, ["LoanAmount", "Loan_Amount_Term"]]
    imp = IterativeImputer(RandomForestRegressor(), max_iter=10, random_state=0)
    df1 = pd.DataFrame(imp.fit_transform(df1), columns=df1.columns)
    df[["LoanAmount", "Loan_Amount_Term"]] = df1

    # 符号化处理
    df['Gender'] = df['Gender'].map({"Male": 0, "Female": 1}).astype(int)
    df['Married'] = df['Married'].map({"No": 0, "Yes": 1}).astype(int)
    df['Dependents'] = df['Dependents'].map({"0": 0, "1": 1, "2": 2, "3+": 3}).astype(int)
    df['Self_Employed'] = df['Self_Employed'].map({"No": 0, "Yes": 1}).astype(int)
    df['Credit_History'] = df['Credit_History'].astype(int)
    df['Property_Area'] = df['Property_Area'].map({"Urban": 0, "Rural": 1, "Semiurban": 2}).astype(int)
    df['Education'] = df['Education'].map({"Not Graduate": 0, "Graduate": 1}).astype(int)

    # feature engineering
    df["LoanAmount_log"] = np.log(df["LoanAmount"])
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Total_Income_log"] = np.log(df["Total_Income"])
    df.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income'], axis=1)
    df.drop(['Gender'], inplace=True, axis=1)

    df['Loan_Status'] = df['Loan_Status'].map({"N": 0, "Y": 1}).astype(int)
    df = df[["Married", "Dependents", "Education", "Self_Employed", "Loan_Amount_Term", "Credit_History",
             "Property_Area", "LoanAmount_log", "Total_Income_log", "Loan_Status"]]
    # df.to_csv('loan2.csv')
    return df