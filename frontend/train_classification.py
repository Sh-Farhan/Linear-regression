import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib

print("1. Loading data...")
df = pd.read_csv(r'D:\CORE-ML\Regression\PROJECT\diabetes_dataset.csv')

# Select our 4 features and the target
X = df[['glucose_fasting', 'bmi', 'age', 'systolic_bp']] 
y = df['diagnosed_diabetes']

# Optional but good practice: handle any missing values just in case
imputer = SimpleImputer(strategy='mean')
X_cleaned = imputer.fit_transform(X)

print("2. Training classification model (this might take a few seconds)...")
model = LogisticRegression(max_iter=1000) 
model.fit(X_cleaned, y)

print("3. Saving the model...")
joblib.dump(model, 'diabetes_clf_model.pkl')
print("✅ DONE! 'diabetes_clf_model.pkl' has been created.")