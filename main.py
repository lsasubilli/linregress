import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn import linear_model

filepath = 'car_dekho.csv'
car_data = pd.read_csv(filepath)

x_train = car_data[['Age']]
y_train = car_data[['Selling_Price']]

regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

predicted = regr.predict(x_train)

plt.scatter(x_train,y_train)
plt.plot(x_train, predicted)
plt.xlabel("Age")
plt.ylabel("Selling Price")
plt.show()
print(r2_score(predicted))
print(regr.coef_)
print(regr.intercept_)
