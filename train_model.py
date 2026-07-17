import pandas as pd
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Keep only required columns
df = df.iloc[:, :2]

# Rename columns
df.columns = ["label", "message"]

# Convert labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


# Apply cleaning
df["message"] = df["message"].apply(clean_text)

# Convert text into numerical features
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["message"])
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel saved successfully.")