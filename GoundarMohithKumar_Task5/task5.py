# ===============================
# Sales Prediction
# Oasis Infobyte - Task 5
# ===============================

# --------- Import Libraries ----------
 
import pandas as pd
import numpy as np

# ----------  Load Dataset -----------

df=pd.read_csv("C:/Users/gound/OneDrive/Desktop/OASIS/GoundarMohithKumar_Task5/Advertising.csv")
print(df.head())

# --------- CLean Dataset ----------

df = df.drop(columns=["Unnamed: 0"])
print(df.head())

# ---------- Define Features & Target ---------- 

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# ---------- Train - Test Split ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# -------- Train Model ---------

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# ---------- Predict --------- 

y_pred = model.predict(X_test)

print(y_pred[:5])

# --------- Evaluate Model ---------

from sklearn.metrics import r2_score, mean_absolute_error

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# ---------- Visualize Results -------

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# ---------- Predict New Sales ----------

sample = [[200, 30, 10]]  # TV, Radio, Newspaper

prediction = model.predict(sample)

print("Predicted Sales:", prediction[0])