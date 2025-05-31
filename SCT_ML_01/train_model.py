import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load dataset
df = pd.read_csv("House Price India.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Show available columns (for debug)
print("Available columns:", df.columns.tolist())

# Drop missing values
df.dropna(inplace=True)

# Select features and target
X = df[[
    "grade of the house",        # overall quality
    "living area",               # above ground living area
    "area of the basement",      # basement area
    "number of bathrooms",       # full bath count
    "built year"                 # year built
]]
y = df["price"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Save the model
with open("house_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully")
