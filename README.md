# Introduction
This project contains most(<b>not all</b>) of the information related to Linear Regression Model. It contains some projects that helps understand how Linear Regression model is used to predict/forecast a value based on input data.  But before getting into the linear regression model let's revise some basic concept of Machine Learning.

## What is Machine Learning Model?
There are different ways we can define the term "Machine Learning Model", some of them are:
- A machine learning model is a program that can find patterns or make decisions from a previously unseen dataset.
- A machine learning model is a methematical representation of a real world process that a computer can use to make predictions or decision based on input data. It is trained on historical data to learn patterns, relationships, and trends, which it can then use to make predictions or decisions on new, unseen data.

## What is Machine learning Algorithm?
A machine learning algorithm is a mathematical method to find patterns in a set of data. Machine Learning algorithms are often drawn from statistics, calculus and linear algebra. Some popular example of machine learning algorithms include linear regression, decision trees, random forest amd XGBoost.

## What is Model Training in machine learning?
The process of running a machine learning algorithm on a dataset (called training data) and optimizing the algorithm to find certain patterns or outputs is called model training.

## What are the different types of Machine Learning?
In general, most machine learning techniques can be classified into supervised learning, unsupervised learning, and reinforcement learning.

# Linear Regression
Linear Regression is a statistical method used for modeling the relationship between a dependent variable and one or more independent variables. It is one of the simplest and most widely used techniques in machine learning and statistical modeling.

Linear regression is a <b>Supervised Learning Model</b>. In supervised learning, the algorithm learns from labeled data, where each example in the dataset is associated with a target variable (or label). The goal is to learn mapping from the input features to the target variable so that the model can make predictions on new, unseen data.

Linear Regression can be classified into two main types based on the number of independent variables:
  1. Simple Linear Regression : Involves one independent variable.
  2. Multiple Linear Regression: Involve more than one independent variable.

Linear regression is widely used for various purposes, including:
  1. Predictive Modeling : Predicting future values of a dependent variable based on historical data.
  2. Understanding relationships: Investigating the relationships between independent and dependent variables.
  3. Inference: Drawing conclusions about the population based on the sample data.

## Steps for creating Linear Regression Model 
* <b>Step 1</b>: Data Preparation
    * Import data
    * View data
    * Check the summary of data
    * If the data variables have missing values, do missing value imputation
    * Missing value imputation: First look at the histogram of each of the data variables
        * If the histogram shows normal distribution, replace the missing values with mean (only for continuous variable)
        * In case, it shows the skewed distribution, replace the missing values with median (only for continuous variable)
        * If the variable is categorical, do the imputation with mode.
        * We can even make a predictive model to impute missing data in a variable. Here, we will treat the variable having missing data as target variable and the other variable as predictors. We’ll divide the data into 2 sets – one without any missing value for that variable and the other with missing values for that variable. The former set would be used as training set to build the predictive model and it would then be applied to the latter set to predict the missing values.
* <b>Step 2</b>: Data Manipulation
    * For categorical independent variables, one can go for many types of encodings – people often choose dummy variables creation for ordinal variables – categorical variables with some order. For a categorical variable having n categories, there should be n-1 dummy variables since the left one variable is taken care of by the intercept.
    * There is also one more type of encoding – One Hot encoding, each category of a categorical variable is converted into a new binary column (1/0).
* <b>Step 3</b>: Univariate Analysis
    * Now, do outlier manipulation for each independent as well as dependent variable. Outliers lead to multi-collinearity and also deteriorate the model. For doing outlier manipulation, one can choose the threshold as +/- 4 percentile since it doesn’t lead to severe loss of the data.
    * Check the distribution of each of the independent variables. If its skewed, it must be transformed. If it’s normal, no transformation required!
* <b>Step 4</b>: Bivariate Analysis
    * Now, go for seeing the relationship between the dependent and each of the independent variables one-by-one. Scatter plot is the best means for that. This lets the person know about the relationship between them. If the relationship is linear, good, if its curvilinear, go for log transformation!
* <b>Step 5</b>: Linear Regression Analysis
    * Go for correlation analysis first. See if the independent variables have high correlation among each-other or not. If they have, it can lead to multi-collinearity! They should be having high correlation with dependent variable but not among themselves!
    * Check the multi-collinearity: Use the variance inflation factor (vif function) to achieve this. It gives the multi-collinearity value only. If it’s less than 5 for a variable keep it in the model, else discard it or exclude it from the model.
    * Now, make a model for the new list of variables, and check vif again!
    * If you want to eliminate this list of steps then use step function directly. It does forward/backward propagation plus taken care of multicollinearity also.
    * Discard the variables having p-value higher than .05, they are least significant variables.
* <b>Step 6</b>: Model Evaluation
    * Evaluation Metrics : Mean Absolute Error, Mean Squared Error and Root Mean Squared Error
    * After getting the final model from the previous steps, go for checking whether or not the model satisfies the assumptions of linear model.
    * Assumptions of Linear Regression:
        * The relationship between independent and dependent variables must be linear. Check this by means of the scatterplots.
        * The residuals should be normally distributed. Residuals = Observed value – Predicted value (Fitted value)
        * Multicollinearity should not be present. Calculate the get an indication about the multi-collinearity values.
        * Homoscedasticity must be present, i.e., the relationship between the residuals and response variable (Predicted variable) should be uniform.
    * Checking if the model satisfies the assumptions
        * Autocorrelation test: Use Durbin Watson Test on the model.
            * The Durbin Watson statistic is a test for autocorrelation in a data set.
            * The DW statistic always has a value between zero and 4.0.
            * A value of 2.0 means there is no autocorrelation detected in the sample. Values from zero to 2.0 indicate positive autocorrelation and values from 2.0 to 4.0 indicate negative autocorrelation.
            * Autocorrelation can be useful in technical analysis, which is most concerned with the trends of security prices using charting techniques in lieu of a company’s financial health or management.
            * For our purposes, a value less than 2 is generally preferred.
            * DW = 2(1-r) where r is the correlation value.
        * Checking normality of errors: Go for seeing the histogram of residuals. Residuals should be normally distributed. This can be checked by visualizing Q-Q Normal plot. If points lie exactly on the line, it is perfectly normal distribution. However, some deviation is to be expected, particularly near the ends, but the deviations should be small.
        * Homoscedasticity: Check the scatterplot between the residuals and response variable. It should be uniform.
            * Check cook’s distance. Observations having high cook’s distance values should be removed and model should be remade.
* <b>Step 7</b>: Validating the model
    * K-fold cross Validation: to calculate the average of k recorded errors also known as cross-validation error. It serves as a performance metric for the model.
    * Using regularized regression models : to handle the correlated independent variables well and to overcome overfitting.
            * Ridge penalty shrinks the co-efficients of correlated predictors towards each-other
            * Lasso tends to pick one of a pair of correlated features and discard the other.
            * The tuning parameter lambda controls the strength of the penalty.
    * Using regressive random forests to carry out regression
    * Boosting: To improve the accuracy of the model
* <b>Step 8</b>: Prediction by linear regression model!

## Steps for creating Linear Regression Model using Python
Sure, here are the steps to perform linear regression in Python using libraries like NumPy, pandas, and scikit-learn:

1. **Import Libraries:**
   ```python
   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import mean_squared_error, r2_score
   ```

2. **Prepare Data:**
   Load your dataset into a pandas DataFrame. Make sure your data is clean and formatted correctly. Separate your features (independent variables) and target variable (dependent variable).
   ```python
   # Assuming 'df' is your DataFrame
   X = df[['feature1', 'feature2', ...]]  # Features
   y = df['target']  # Target variable
   ```

3. **Split Data into Training and Testing Sets:**
   Split your data into training and testing sets to evaluate the performance of the model.
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

4. **Create a Linear Regression Model:**
   ```python
   model = LinearRegression()
   ```

5. **Train the Model:**
   Fit the model to the training data.
   ```python
   model.fit(X_train, y_train)
   ```

6. **Make Predictions:**
   Use the trained model to make predictions on the testing data.
   ```python
   y_pred = model.predict(X_test)
   ```

7. **Evaluate the Model:**
   Assess the performance of the model using evaluation metrics such as Mean Squared Error (MSE) and R-squared.
   ```python
   mse = mean_squared_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)
   ```

8. **Interpret Results:**
   Analyze the coefficients of the model to understand the relationship between the features and the target variable.
   ```python
   coefficients = model.coef_
   intercept = model.intercept_
   ```

9. **Visualization (Optional):**
   Visualize the relationship between the features and the target variable using plots.
   ```python
   # Example for a single feature linear regression
   import matplotlib.pyplot as plt
   plt.scatter(X_test, y_test, color='black')
   plt.plot(X_test, y_pred, color='blue', linewidth=3)
   plt.xlabel('Feature')
   plt.ylabel('Target')
   plt.show()
   ```

These are the basic steps for performing linear regression in Python. You can further enhance your model by adding more features, handling outliers, or trying different regression techniques.


## Evaluating Linear Regression Model
Evaluating a linear regression model is crucial to understand its performance and reliability in predicting outcomes. Here are some common evaluation techniques for linear regression models:

1. **Mean Squared Error (MSE)**:
   - Calculates the average squared difference between the predicted values and the actual values.
   - Provides a measure of the model's accuracy, where lower values indicate better performance.
   - Computed as:
     \[ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]
   where \( n \) is the number of observations, \( y_i \) is the actual value, and \( \hat{y}_i \) is the predicted value for observation \( i \).

2. **Root Mean Squared Error (RMSE)**:
   - The square root of the MSE, providing an interpretable measure in the same units as the target variable.
   - Gives a sense of the average magnitude of the errors.
   - Computed as:
     \[ \text{RMSE} = \sqrt{\text{MSE}} \]

3. **R-squared (R²)**:
   - Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
   - Ranges from 0 to 1, where 1 indicates a perfect fit.
   - Computed as:
     \[ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} \]
   where \( \bar{y} \) is the mean of the observed data.

4. **Adjusted R-squared**:
   - A modified version of R-squared that adjusts for the number of predictors in the model.
   - Penalizes excessive use of variables that do not improve the model's performance.
   - Provides a better indication of the model's goodness of fit when comparing models with different numbers of predictors.

5. **Residual Analysis**:
   - Examination of the residuals (the differences between observed and predicted values).
   - Plotting residuals against predicted values or independent variables to check for patterns.
   - Ideally, residuals should be randomly distributed around zero with constant variance.

6. **Durbin-Watson Statistic**:
   - Measures autocorrelation in the residuals.
   - Values close to 2 indicate no autocorrelation, while values significantly different from 2 may indicate autocorrelation.

7. **Feature Importance**:
   - Assessing the significance of each independent variable's contribution to the model.
   - Using coefficients, p-values, or feature importance scores to understand variable importance.

By using a combination of these evaluation techniques, you can gain a comprehensive understanding of your linear regression model's performance and identify areas for improvement.
