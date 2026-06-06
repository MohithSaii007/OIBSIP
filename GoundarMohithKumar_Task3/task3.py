# ===============================
# Car Price Prediction
# Oasis Infobyte - Task 3
# ===============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ----------- Load Dataset -----------

df = pd.read_csv("C:/Users/gound/OneDrive/Desktop/OASIS/GoundarMohithKumar_Task3/car data.csv")

print("First 5 Rows:")
print(df.head())

# -----------  Remove Car_Name column ----------


df = df.drop("Car_Name", axis=1)

# --------- Convert categorical columns -----------
df = pd.get_dummies(df, drop_first=True)

#  ----------  Features and Target -------------
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# ---------  Split Dataset ---------- 


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -------- Train Model ----------


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------  Prediction -----------

y_pred = model.predict(X_test)

# -----------  Evaluation ------------


r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("/nModel Performance")
print("R2 Score:", r2)
print("Mean Absolute Error:", mae)

# --------  Example Prediction ------------


sample = X_test.iloc[[0]]
predicted_price = model.predict(sample)

print("/nPredicted Price:", predicted_price[0])
print("Actual Price:", y_test.iloc[0])