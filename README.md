# BMW Used Car Price Prediction Model

<p align="center">

  <a href="https://upload.wikimedia.org/wikipedia/commons/4/44/BMW.svg">
      <img src="https://upload.wikimedia.org/wikipedia/commons/4/44/BMW.svg" width="100"/>
  </a>
</p>

## **Project Overview**

This project aims to predict the price of used BMW cars using a dataset of real BMW cars sold in 2018. The dataset provides various features about each car, including mileage, engine power, fuel type, and registration date, which are used to train a regression model. The goal is to build a model that accurately estimates the price of a used car based on its specifications.

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

- **Machine Learning Model**: Built using Python's popular libraries (e.g., Scikit-learn, Pandas, NumPy).
- **Data Preprocessing**: Includes data cleaning, feature engineering, and normalization.
- **Model Training**: Implemented the linear regression machine learning model (more types of models will be trained in the future).
- **Evaluation**: Assessed models using RMSE, MAE, and R-squared metrics.
- **Prediction**: Provides accurate car price estimates based on car specifications.

## **Project Structure**

```bash
BMW-Car-Price-Prediction-Model/
├── bmw_pricing_challenge.csv     # Dataset
├── main.py                       # Data Preprocessing, Filtering, Key Features Outlining and Model Training
└── README.md                     # Project overview and instructions
```


## **Objective**
The goal of this project is to:

 - Analyze the dataset and explore the relationships between different features and the car price.
 - Train a regression model to predict the price of a used BMW car based on its specifications.
 - Evaluate the model's performance using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and R-squared (R²).


## **Model Training**
The project uses a Linear Regression model to predict the price of the cars. The steps involved include:

- **Data Cleaning:** Handling missing values, removing duplicates, and converting non-numeric features to numeric ones.
- **Feature Engineering:** Adding useful features like time_to_sale (days between car registration and auction) and extracting the year from the registration_date.
- **Correlation Analysis:** Identifying which features are most correlated with the price.
- **Model Evaluation:** Using metrics like MSE, RMSE, MAE, MAPE, and R² to evaluate model performance.

  
**Key Findings:**
- **Most Correlated Features:** A correlation analysis shows that certain features have a higher impact on the price.

- **Model Performance:** The model achieved the following results:

    **MSE:** 20,827,900.99
  
    **RMSE:** 4563.76
    
    **MAE:** 2953.36
  
    **MAPE:** 73.44%     *(Relatively good R² Score compared to the other submissions of the price challenge)*
  
    **R² Score:** 0.723   
  




## **Future Work**
- **Feature Improvement:** Explore more feature engineering techniques to capture hidden patterns in the data.
- **Advanced Models:** Experiment with more complex models like Random Forest, Gradient Boosting, or Neural Networks to improve prediction accuracy.
- **Hyperparameter Tuning:** Perform grid search or random search to find the best model hyperparameters.
- **Interactive Price Prediction Feature:** A planned enhancement to the project is the development of a command-line interface (CLI) that allows users to input specific car features (such as mileage, engine power, and registration date) directly into the script. This feature will enable real-time price predictions based on user-defined specifications, making the model more accessible and user-friendly.


  
