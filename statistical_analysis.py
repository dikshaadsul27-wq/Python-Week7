import pandas as pd

#Descriptive Statistics

# Load dataset
df = pd.read_csv("customer_churn.csv")

# Descriptive statistics
desc_stats = df[['Tenure','MonthlyCharges','TotalCharges']].describe()
print(desc_stats)

# Mode calculation
for col in ['Tenure','MonthlyCharges','TotalCharges']:
    print(f"Mode of {col}: {df[col].mode()[0]}")

#Data Distribution Analysis

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# Histogram for MonthlyCharges
sns.histplot(df['MonthlyCharges'], kde=True)
plt.title("Monthly Charges Distribution")
plt.show()

# Normality test
stat, p = shapiro(df['MonthlyCharges'])
print("Shapiro-Wilk Test p-value:", p)

#Correlation Analysis

# Pearson correlation
corr = df[['Tenure','MonthlyCharges','TotalCharges','Churn']].corr()

# Heatmap
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#Hypothesis Testing

from scipy.stats import ttest_ind, chi2_contingency

# T-test: MonthlyCharges between churn vs non-churn
churn_yes = df[df['Churn']==1]['MonthlyCharges']
churn_no = df[df['Churn']==0]['MonthlyCharges']
t_stat, p_val = ttest_ind(churn_yes, churn_no)
print("T-test MonthlyCharges (Churn vs Non-Churn):", p_val)

# Chi-square: Contract vs Churn
contract_churn = pd.crosstab(df['Contract'], df['Churn'])
chi2, p, dof, ex = chi2_contingency(contract_churn)
print("Chi-square Contract vs Churn p-value:", p)

# Chi-square: PaymentMethod vs Churn
payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'])
chi2, p, dof, ex = chi2_contingency(payment_churn)
print("Chi-square PaymentMethod vs Churn p-value:", p)

#Confidence Intervals

import numpy as np
import scipy.stats as st

# 95% CI for MonthlyCharges
mean_mc = np.mean(df['MonthlyCharges'])
sem_mc = st.sem(df['MonthlyCharges'])
ci_mc = st.t.interval(0.95, len(df['MonthlyCharges'])-1, loc=mean_mc, scale=sem_mc)
print("95% CI for MonthlyCharges:", ci_mc)

#Logistic Regression

import statsmodels.api as sm

# Encode categorical variables
df_encoded = pd.get_dummies(df[['Tenure','MonthlyCharges','Contract','PaymentMethod','Churn']], drop_first=True)

# Ensure all columns are numeric
X = df_encoded.drop('Churn', axis=1).astype(float)
y = df_encoded['Churn'].astype(int)

# Logistic regression
logit_model = sm.Logit(y, sm.add_constant(X)).fit()
print(logit_model.summary())