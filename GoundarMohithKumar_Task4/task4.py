# ===============================
# Email Spam Detection Project
# Oasis Infobyte - Task
# ===============================

import pandas as pd

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

print(df.head())

# Keep only useful columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

print(df.head())

# ===============================
# Convert labels
# spam = 1, ham = 0
# ===============================
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ===============================
# Text processing
# ===============================
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(df['message'])
y = df['label']

# ===============================
# Train test split
# ===============================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ===============================
# Model Training (Naive Bayes BEST for spam)x
# ===============================
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)

# ===============================
# Prediction
# ===============================
y_pred = model.predict(X_test)

# ===============================
# Accuracy
# ===============================
from sklearn.metrics import accuracy_score, classification_report

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# ===============================
# Test custom message
# ===============================
sample = ["Congratulations! You won a free iPhone. Click now"]

sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)

print("\nSample Prediction:", "SPAM" if prediction[0] == 1 else "HAM")