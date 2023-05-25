#zu großer datensatz dafür
import pandas as pd
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.tree import plot_tree
loandata= pd.read_excel(r"C:\Users\fedey\Desktop\Bachelorarbeit\empirie\Ausreisergefiltert.xlsx")
print(loandata)
loandata = loandata[['Gender','NewCreditCustomer','Ruckzahlungsquote' ]]
print(loandata)
x = loandata.drop(columns ='Ruckzahlungsquote')
y= loandata['Ruckzahlungsquote']
x_train,x_test,y_train,y_test, = train_test_split(x,y, test_size = 0.3, random_state = 23)
imputer = SimpleImputer()
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)
dtr = DecisionTreeRegressor()
dtr.fit(x_train_imputed, y_train)

y_pred_train = dtr.predict(x_train_imputed)
print(y_pred_train)
plt.scatter(y_train, y_pred_train)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.show()

y_pred_test = dtr.predict(x_test_imputed)
plt.scatter(y_test, y_pred_test)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.show()
r2 = r2_score(y_train,y_pred_train)
print(r2)
score = dtr.score(x_test_imputed, y_test)
print(f"Accuracy: {score}")

plt.figure(figsize=(20,20))
plot_tree(dtr, filled=True, feature_names=x.columns)
plt.savefig('decision_tree.png')
plt.show()
