import os
#importing files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

store_sales = pd.read_csv("train.csv")
store_sales.head(10)

#Check For Null Values in the Dataset
store_sales.info()

#Dropping Store and Item Columns
store_sales = store_sales.drop(['store','item'], axis=1)
store_sales.info()

#Converting Date from object DT to datetime DT
store_sales['date'] = pd.to_datetime(store_sales['date'])
store_sales.info()

#Converting the Date to Month period and then summing the nummber of items in each month
store_sales['date'] = store_sales['date'].dt.to_period("M")
monthly_sales = store_sales.groupby('date').sum().reset_index()

#Coverting the resulting date column to datestamp datatype
monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
monthly_sales.head(10)

#Visualisation
plt.figure(figsize=(15,5))
plt.plot(monthly_sales['date'],monthly_sales['sales'])
plt.ylabel("Sales")
plt.xlabel("Date")
plt.title("Monthly Customer Sales")
plt.show()

#Call the difference on the sales column to make data stationary
monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
monthly_sales = monthly_sales.dropna()
monthly_sales.head(10)

plt.figure(figsize=(15,5))
plt.plot(monthly_sales['date'],monthly_sales['sales'])
plt.ylabel("Sales")
plt.xlabel("Date")
plt.title("Monthly Customer Sales Difference")
plt.show()

#Dropping off Sales and Date
supervised_data = monthly_sales.drop(['sales','date'],axis=1)

#Preparing the supervised data
for i in range(1,13):
  col_name = 'month_' + str(i)
  supervised_data[col_name] = supervised_data['sales_diff'].shift(i)
supervised_data = supervised_data.dropna().reset_index(drop = True)
supervised_data.head(10)

#Split The data into Train and Test Data
train_data = supervised_data[:-12]
test_data = supervised_data[-12:]
print("Train Data Shape :",train_data.shape)
print("Test Data Shape :",test_data.shape)

scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

x_train , y_train = train_data[:,1:] , train_data[:,0:1]
x_test , y_test = test_data[:,1:] , test_data[:,0:1]
y_train = y_train.ravel()
y_test = y_test.ravel()
print("X train shape :",x_train.shape)
print("Y train shape :",y_train.shape)
print("X Test Shape",x_test.shape)
print("Y Test Shape",y_test.shape)

# Make Prediction Data Frame to Merge to merge the predicted sales of all trained algorithms
sales_date = monthly_sales['date'][-12:].reset_index(drop = True)
predict_df = pd.DataFrame(sales_date)

act_sales = monthly_sales['sales'][-13:].to_list()
print(act_sales)

#To create Linear Reagression Model and predicted output
lr_model = LinearRegression()
lr_model.fit(x_train,y_train)
lr_pre = lr_model.predict(x_test)

lr_pre = lr_pre.reshape(-1,1)
#This is a set matrix - contains Input features of the test data and the predicted output
lr_pre_test_set = np.concatenate([lr_pre,x_test],axis =1)
lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)

result_list = []
for index in range(0,len(lr_pre_test_set)):
  result_list.append(lr_pre_test_set[index][0]+ act_sales[index])
lr_pre_series = pd.Series(result_list, name="Linear Prediction")
predict_df = predict_df.merge(lr_pre_series, left_index= True, right_index= True)

print(predict_df)

lr_mse = np.sqrt(mean_squared_error(predict_df['Linear Prediction'],monthly_sales['sales'][-12:]))
lr_mae = mean_absolute_error(predict_df['Linear Prediction'],monthly_sales['sales'][-12:])
lr_r2 = r2_score = (predict_df['Linear Prediction'],monthly_sales['sales'][-12:])

print("Linear Regression MSE :",lr_mse)
print("Linear Regression MAE :",lr_mae)
print("Linear Regression R2 :",lr_r2)

#Visualisation of the Prediction against actual Sales
plt.figure(figsize=(15,5))
plt.plot(monthly_sales['date'],monthly_sales['sales'])
plt.plot(predict_df['date'],predict_df['Linear Prediction'])
plt.ylabel("Sales")
plt.xlabel("Date")
plt.title("Customer Sales Forecast Using LR Model")
plt.show()