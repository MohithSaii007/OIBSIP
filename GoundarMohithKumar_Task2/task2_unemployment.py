# ===============================
# Unemployment Analysis in India
# Task 2 - Oasis Infobyte
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Load Dataset ----------
df = pd.read_csv(r"C:/Users/gound/OneDrive/Desktop/OASIS/GoundarMohithKumar_Task2/Unemployment in India.csv")

# ---------- Clean columns ----------
df.columns = df.columns.str.strip()

print("\nColumns:")
print(df.columns)

print("\nFirst rows:")
print(df.head())

# ---------- Check missing values ----------
print("\nMissing values:")
print(df.isnull().sum())

# ===============================
# 1. Unemployment Rate by Area
# ===============================

plt.figure(figsize=(10,5))

sns.barplot(
    data=df,
    x="Area",
    y="Estimated Unemployment Rate (%)"
)

plt.title("Unemployment Rate by Area")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===============================
# 2. Average Unemployment by Area
# ===============================

area_avg = df.groupby("Area")["Estimated Unemployment Rate (%)"].mean().reset_index()

plt.figure(figsize=(8,5))

sns.barplot(
    data=area_avg,
    x="Area",
    y="Estimated Unemployment Rate (%)"
)

plt.title("Average Unemployment by Area")
plt.tight_layout()
plt.show()

# ===============================
# 3. Employment vs Labour Participation
# ===============================

plt.figure(figsize=(10,5))

sns.scatterplot(
    data=df,
    x="Estimated Employed",
    y="Estimated Labour Participation Rate (%)",
    hue="Area"
)

plt.title("Employment vs Labour Participation Rate")
plt.tight_layout()
plt.show()

# ===============================
# 4. Correlation Heatmap
# ===============================

plt.figure(figsize=(6,4))

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")

plt.title("Correlation Between Variables")
plt.show()