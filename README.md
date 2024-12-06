# BMW Used Car Price Prediction Model

<p align="center">

  <a href="https://upload.wikimedia.org/wikipedia/commons/4/44/BMW.svg">
      <img src="https://upload.wikimedia.org/wikipedia/commons/4/44/BMW.svg" width="100"/>
  </a>
</p>

## **Project Overview**

This project aims to predict the price of used BMW cars using a dataset of real BMW cars sold in 2018. The dataset provides various features about each car, including mileage, engine power, fuel type, and registration date, which are used to train regression models. The goal is to build a model that accurately estimates the price of a used car based on its specifications.

## **Dataset**
The dataset consists of approximately 5,000 BMW cars sold in 2018. It includes the following columns:

**mileage:** The total distance the car has traveled (in kilometers).

**engine_power:** The power of the engine (in kW).

**registration_date:** The date the car was first registered.

**fuel:** Type of fuel used (diesel, petrol).

**paint_color:** The exterior color of the car.

**car_type:** Body style of the car (sedan, estate, convertible, etc.).

**feature_1 to feature_8:** Additional equipment features of the car.

**price:** The target variable, representing the price the car was sold at.


## **Features**

- **Machine Learning Models**: Experimented with multiple models, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, XGBoost, Support Vector Regressor (SVR), and TensorFlow-based models.
- **Data Preprocessing**: Includes data cleaning, feature engineering, normalization with `StandardScaler`, and encoding categorical variables using `LabelEncoder`.
- **Model Training**: Implemented and compared different regression models for improved accuracy.
- **Evaluation**: Assessed models using RMSE, MAE, MAPE, and R-squared metrics.
- **Prediction**: Provides accurate car price estimates based on car specifications.

## **Project Structure**

```bash
BMW-Car-Price-Prediction-Model/
├── bmw_pricing_challenge.csv     # Dataset
├── BMW_Car_Price_Predictor.py    # Data Preprocessing, Filtering, Key Features Outlining and Model Training
└── README.md                     # Project overview and instructions
```


## **Objective**
The goal of this project is to:

 - Analyze the dataset and explore the relationships between different features and the car price.
 - Train and compare regression models to predict the price of a used BMW car based on its specifications.
 - Evaluate the models' performance using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and R-squared (R²).


## **Model Training**
The project now includes a diverse range of models to improve prediction accuracy. The steps involved include:

- **Data Cleaning:** Handling missing values, removing duplicates, and converting non-numeric features to numeric ones.
- **Feature Engineering:** Adding useful features like time_to_sale (days between car registration and auction) and extracting the year from the registration_date.
- **Feature Scaling:** Normalizing data using `StandardScaler` to improve model performance.
- **Correlation Analysis:** Identifying which features are most correlated with the price.
- **Advanced Models:** Training and evaluating models such as Decision Tree Regressor, Random Forest Regressor, XGBoost, and TensorFlow-based Neural Networks in addition to Linear Regression.
- **Model Evaluation:** Using metrics like MSE, RMSE, MAE, MAPE, and R² to evaluate model performance.

  
**Key Findings:**
- **Most Correlated Features:** A correlation analysis shows that certain features have a higher impact on the price.

- **Model Performance:** Below are the performance results of key models:

    **Linear Regression**:
    - **MSE:** 19,671,873.87
    - **RMSE:** 4,435.30
    - **MAE:** 3,062.47
    - **MAPE:** 67.28%
    - **R² Score:** 0.7204

    **Decision Tree Regressor**:
    - **MSE:** 20,210,916.61
    - **RMSE:** 4,495.66
    - **MAE:** 2,885.87
    - **MAPE:** 63.77%
    - **R² Score:** 0.7128

    **Random Forest Regressor (Best Performance)**:
    - **MSE:** 11,042,179.05
    - **RMSE:** 3,322.98
    - **MAE:** 2,099.72
    - **MAPE:** 52.91%
    - **R² Score:** 0.8431

    **XGBoost**:
    - **MSE:** 11,570,707.58
    - **RMSE:** 3,401.57
    - **MAE:** 2,148.32
    - **MAPE:** 54.58%
    - **R² Score:** 0.8356

    **Neural Network**:
    - **MSE:** 13,669,512.14
    - **RMSE:** 3,697.23
    - **MAE:** 2,445.35
    - **MAPE:** 53.90%
    - **R² Score:** 0.8057


## **Future Work**
- **Feature Improvement:** Explore more feature engineering techniques to capture hidden patterns in the data.
- **Hyperparameter Tuning:** Perform grid search or random search to find the best model hyperparameters.
- **Interactive Price Prediction Feature:** A planned enhancement to the project is the development of a command-line interface (CLI) that allows users to input specific car features (such as mileage, engine power, and registration date) directly into the script. This feature will enable real-time price predictions based on user-defined specifications, making the model more accessible and user-friendly.



  
