# %%
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

#Graphing Libraries
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



# %%
data=pd.read_csv('bmw_pricing_challenge.csv')
X=data.drop(columns=['price'])
Y=data['price'] #target variable

# %%

# Cleaning Data

data.drop(['maker_key','model_key'],axis=1,inplace=True)
data.drop_duplicates(inplace=True)
data = data.dropna()

data.head()



# %%
# missing values
data.isnull().sum().sum()


# %%

# Add days between registration date and sold_at
data['sold_at'] = pd.to_datetime(data['sold_at'])
data['registration_date'] = pd.to_datetime(data['registration_date'])
data['time_to_sale'] = data['sold_at'] - data['registration_date']

# Add registration year
data['year'] = data['registration_date'].dt.year
data
# %%

# Convert non-numeric values ito numeric values
enc = LabelEncoder()
for col in data.columns:
    if data[col].dtypes != 'int64':
        data[col]= enc.fit_transform(data[col])


data
# %%
data.columns

# %%
data.describe()


# %%
data.dtypes[data.dtypes!='object']


# %%
pd.DataFrame(data.isnull().sum().sort_values(ascending=False)).head(20)



###Data Visualization:
# %%

#Price Distribution
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
sns.histplot(data['price'], kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')

# %%
#Correlation Heatmap
plt.subplot(2, 3, 2)
corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')

# %%
#Mileage vs. Price Scatter Plot
plt.subplot(2, 3, 3)
sns.scatterplot(x=data['mileage'], y=data['price'])
plt.title('Mileage vs. Price')
plt.xlabel('Mileage')
plt.ylabel('Price')

# %%
#Engine Power vs. Price Scatter Plot
plt.subplot(2, 3, 4)
sns.scatterplot(x=data['engine_power'], y=data['price'])
plt.title('Engine Power vs. Price')
plt.xlabel('Engine Power')
plt.ylabel('Price')

# %%
#Fuel Type Distribution
plt.subplot(2, 3, 5)
sns.countplot(data=data, x='fuel')
plt.title('Fuel Type Distribution')
plt.xlabel('Fuel Type')
plt.ylabel('Count')





# %%

# Which features are most correlated to the price?

price_corr = data.corrwith(data.price).reset_index().rename(columns={'index': 'features', 0: 'values'})
price_corr['values'] = price_corr['values'].apply(lambda x : abs(x))
price_corr.sort_values(by='values', inplace=True)
price_corr.style.background_gradient(cmap='viridis')



# %%

# Predicting Price:

data_score=[]
price = data['price']
data_price= price.values.reshape(-1,1)

from math import sqrt

def regression(y_test_, y_pred_, print_ =False):
    mse=mean_squared_error(y_test_, y_pred_)
    rmse= sqrt(mse)
    mae= mean_absolute_error(y_test_, y_pred_)
    mape=mean_absolute_percentage_error(y_test_, y_pred_)
    r2=r2_score(y_test_, y_pred_)

    if print_:
        print(f"mse: {mse}")
        print(f"rmse: {rmse}")    
        print(f"mae: {mae}")
        print(f"mape: {mape}")
        print(f"r2_score {r2}")

    return r2

def calculate_linear_regression(X, y, print_ =False):
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    linear_regression= LinearRegression()
    linear_regression.fit(x_train, y_train) #training
    y_pred= linear_regression.predict(x_test)

    return regression(y_test,y_pred, print_)



# %%
X= data.drop('price', axis=1).copy()
y= data.price
calculate_linear_regression(X,y,True)
