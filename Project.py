import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
from flask import Flask, request, jsonify

df = pd.read_csv('Salary Prediction of Data Professions.csv')

print(df.head())

print(df.describe())

print(df.isnull().sum())

df['DOJ'] = pd.to_datetime(df['DOJ'])
df['CURRENT DATE'] = pd.to_datetime(df['CURRENT DATE'])

plt.figure(figsize=(10, 6))
sns.histplot(df['SALARY'], kde=True)
plt.title('Distribution of Salaries')
plt.show()

# Categorize Age: Create age groups
age_bins = [20, 30, 40, 50, 60, 70]
age_labels = ['20-29', '30-39', '40-49', '50-59', '60-69']
df['AGE_GROUP'] = pd.cut(df['AGE'], bins=age_bins, labels=age_labels, right=False)

# Interaction Features: Experience * Ratings
df['EXP_RATING'] = df['PAST EXP'] * df['RATINGS']

# Ratios and Rates: Leave Utilization Rate
df['LEAVE_UTILIZATION'] = df['LEAVES USED'] / (df['LEAVES USED'] + df['LEAVES REMAINING'])

# Encoding categorical variables
label_encoders = {}
for column in ['SEX', 'DESIGNATION', 'UNIT', 'AGE_GROUP']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Print encoded classes
print("Encoded values for 'sex':")
for class_label, class_name in enumerate(label_encoders['SEX'].classes_):
    print(f"{class_name}: {class_label}")
    
print("Encoded values for 'DESIGNATION':")
for class_label, class_name in enumerate(label_encoders['DESIGNATION'].classes_):
    print(f"{class_name}: {class_label}")

print("\nEncoded values for 'UNIT':")
for class_label, class_name in enumerate(label_encoders['UNIT'].classes_):
    print(f"{class_name}: {class_label}")

# Calculate tenure in years
df['TENURE'] = (df['CURRENT DATE'] - df['DOJ']).dt.days / 365.25

# Dropping columns not needed for modeling
df = df.drop(columns=['FIRST NAME', 'LAST NAME', 'DOJ', 'CURRENT DATE'])

# Visualizing the correlation matrix with a focus on predicting salary
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix Focusing on Salary Prediction')
plt.show()

print(correlation_matrix['SALARY'].sort_values(ascending=False))

from sklearn.preprocessing import StandardScaler

# Handling missing values by filling with mean
df.fillna(df.mean(), inplace=True)

# Splitting features and target variable
X = df[['AGE', 'LEAVES USED', 'LEAVES REMAINING', 'RATINGS', 'PAST EXP', 'TENURE','SEX', 'DESIGNATION', 'UNIT']]
y = df['SALARY']

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Initializing models
lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(random_state=42)
gb = GradientBoostingRegressor(random_state=42)

# Training models
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
gb.fit(X_train, y_train)
rf.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
# Custom accuracy metric: percentage of predictions within 10% of the actual value
    accuracy_within_10_percent = np.mean(np.abs((y_test - y_pred) / y_test) < 0.10) * 100

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R-squared: {r2}')
    print(f'Accuracy within 10%: {accuracy_within_10_percent}%')

# Evaluating each model
print("Linear Regression:")
evaluate_model(lr, X_test, y_test)

print("\nDecision Tree:")
evaluate_model(dt, X_test, y_test)

print("\nRandom Forest:")
evaluate_model(rf, X_test, y_test)

print("\nGradient Boosting:")
evaluate_model(gb, X_test, y_test)

# Assuming your original data loading and preprocessing steps up to scaling
# are the same as before

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert X_scaled back to a DataFrame with original column names
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Define preprocessing steps for numerical and categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['AGE', 'LEAVES USED', 'LEAVES REMAINING', 'RATINGS', 'PAST EXP', 'TENURE']),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['SEX', 'DESIGNATION', 'UNIT'])
    ])

# Complete pipeline with Gradient Boosting
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Training the pipeline
pipeline.fit(X_train, y_train)

# Saving the model using joblib
joblib.dump(pipeline, 'salary_prediction_model.pkl')