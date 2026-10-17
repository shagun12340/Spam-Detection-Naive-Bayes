import json
import string

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def clean_text(text):
    """Same cleaning as main.py: lowercase + remove punctuation."""
    text = str(text).lower()
    return text.translate(str.maketrans("", "", string.punctuation))


# 1. Load data (same file and columns as main.py)
df = pd.read_csv("data/spam.csv", encoding="latin-1").iloc[:, :2]
df.columns = ["label", "message"]
df = df.drop_duplicates().dropna()
df["target"] = (df["label"] == "spam").astype(int)  # 1 = spam, 0 = ham
df["clean"] = df["message"].apply(clean_text)

# 2. Split FIRST, so the test set is never seen during training
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["target"],
    test_size=0.2, random_state=42, stratify=df["target"])

# 2b. Add modern scam examples (job offers, KYC, OTP, loans, parcels...) to
#     the TRAINING data only. The test set stays original, so accuracy is honest.
extra = pd.read_csv("data/extra_scam_data.csv").dropna()
extra_X = extra["message"].apply(clean_text)
extra_y = (extra["label"] == "spam").astype(int)
for _ in range(3):  # repeat so the small extra set has some weight
    X_train = pd.concat([X_train, extra_X], ignore_index=True)
    y_train = pd.concat([y_train, extra_y], ignore_index=True)

# 3. TF-IDF with English stopword removal; fit on training data only
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train Multinomial Naive Bayes
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vec, y_train)

# 5. Evaluate on unseen test data
pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, pred)
p, r, f1, _ = precision_recall_fscore_support(y_test, pred, pos_label=1, average="binary")
cm = confusion_matrix(y_test, pred)

print(f"Accuracy : {acc*100:.2f}%")
print(f"Spam precision: {p:.4f}  recall: {r:.4f}  F1: {f1:.4f}")
print("Confusion matrix [[TN FP], [FN TP]]:\n", cm)
print(classification_report(y_test, pred, target_names=["ham", "spam"]))

# 6. Save model, vectorizer and metrics (the app can read metrics.json)
joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
with open("models/metrics.json", "w") as f:
    json.dump({"accuracy": round(acc * 100, 2), "precision": round(p, 4),
               "recall": round(r, 4), "f1": round(f1, 4),
               "confusion_matrix": cm.tolist(),
               "train_size": len(X_train), "test_size": len(X_test)}, f, indent=2)
