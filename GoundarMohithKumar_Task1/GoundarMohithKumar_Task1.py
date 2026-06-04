#  --------- Loading Dataset ----------

import pandas as pd
df = pd.read_csv("Iris.csv")

# -------- Dataset Information --------

print(df.head())
print(df.info())
print(df.describe())
print(df["Species"].value_counts())

# --------- Visualize Dataset ---------

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue="Species")
plt.show()


# --------- Features and Target --------

X = df.drop(["Id", "Species"], axis=1)

y = df["Species"]


# ---------- Split Dataset ---------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------- Train Model ----------

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)


# ---------- Make Predictions ----------

y_pred = model.predict(X_test)

print(y_pred)


# ---------- Check Accuracy ----------

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# ----------- Predict New Flower ----------

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("Predicted Species:", prediction[0])