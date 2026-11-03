import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("/content/diabetes.csv")

data.head()


data.shape


data.columns.to_list()


data.dtypes


data.isna().sum()


data.describe()


x = data.drop(columns='Outcome', axis=1)
y = data['Outcome']
x.shape


# **Feature variables**
# 
# 
# *   Pregnancies : Number of times pregnant
# *   Glucose: Glucose concentraion which is a direct diabetes
# *   Blood Pressure: Monitors cardiovascular and metabolic health
# *   Skin thickness : Estimates subcutaneous body fat
# 
# *   Insulin : Assess insulin resistance and secretion
# *   BMI : measures overall body mass and obesity levels
# *   DiabetesPedigreeFunction : Heraditory risk
# *   Age : Patient's age
# 
# **Target Variable**
# 
# *   Outcome : Binary output with 0 and 1. 0 indicates No diabetes and 1 indicates Diabetes diagnosed


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.85)
print(x_train.shape)
print(x_test.shape)
cat_columns = x_train.select_dtypes(include='object').columns.to_list()
print("Categorical columns:",cat_columns)


#There are no missing values in the dataset, still verifying
train_medians = x_train.median()
x_train = x_train.fillna(train_medians)
x_test = x_test.fillna(train_medians)
print("Missing values",x_train.isna().sum().sum())


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
feature_names = x.columns
x_train = pd.DataFrame(x_train, columns=feature_names)
x_test = pd.DataFrame(x_test, columns=feature_names)
x_train.head()


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


#Why Logistic Regression?


# Logistic Regression is the standard algorithm for clinical diagnostic data classification.
# It is appropriate because
# 
# 
# 1.   Logistic Regression outputs a continuous probablility score between 0 and 1.
# 2.  The probability of having diabetes can be used for suggesting actions like follow-up lab tests and other procedures
# 3. Logistic Regression trains and makes predictions fastly with low memory. So that it can be used with low resource devices and can be associated with existing clinical systems
# 4. It is simpler and avoids overfitting
# 5. Every feature gets a weight, so it is easy to point on why the prediction was made


from sklearn.metrics import(accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)
y_pred=model.predict(x_train)
print("Accuracy Score:",accuracy_score(y_train,y_pred))
print("Precision Score:",precision_score(y_train,y_pred))
print("Recall Score:",recall_score(y_train,y_pred))
print("F1 Score:",f1_score(y_train,y_pred))


#test accuracy
y_pred=model.predict(x_test)
print("Accuracy Score:",accuracy_score(y_test,y_pred))
print("Precision Score:",precision_score(y_test,y_pred))
print("Recall Score:",recall_score(y_test,y_pred))
print("F1 Score:",f1_score(y_test,y_pred))


confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d',cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual LAbel')
plt.show()


# **Interpretation of model results**
# 
# 
# 1.   Strong negative identification : Model performs well at identifying healthy individuals
# 
# 2.   False negatives : A recall of 50% means half of the diabetic patients are missed under standard probability decision thershold
# 
# 3. The calssification decision threshold can be lowered to capture more true positive diabetes cases and can lower false negatives
# 
# 
# ---
