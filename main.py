import streamlit as st
import pandas as pd
import joblib
import string
import matplotlib.pyplot as plt

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = df.iloc[:, :2]
df.columns = ["label", "message"]

total_messages = len(df)
spam_messages = len(df[df["label"] == "spam"])
ham_messages = len(df[df["label"] == "ham"])

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Spam Detection using Naive Bayes",
    page_icon="📩",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 Project Information")

st.sidebar.markdown("""
### 📩 Spam Detection

**Algorithm**
- Multinomial Naive Bayes

**Vectorizer**
- CountVectorizer

**Dataset**
- SMS Spam Collection

**Accuracy**
- 97.85%
""")

# -----------------------------
# Title
# -----------------------------
st.title("📩 Spam Detection using Naive Bayes")

st.write(
    "This application predicts whether an SMS message is **Spam** or **Not Spam** using Machine Learning."
)

st.success("Model Accuracy: 97.85%")

# -----------------------------
# Dataset Statistics
# -----------------------------
st.header("📊 Dataset Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Messages", total_messages)

with col2:
    st.metric("Spam Messages", spam_messages)

with col3:
    st.metric("Not Spam", ham_messages)

# -----------------------------
# Pie Chart
# -----------------------------
st.subheader("Spam vs Not Spam")

fig, ax = plt.subplots(figsize=(5, 5))

ax.pie(
    [spam_messages, ham_messages],
    labels=["Spam", "Not Spam"],
    autopct="%1.1f%%",
    startangle=90
)

ax.set_title("Dataset Distribution")

st.pyplot(fig)

st.divider()

# -----------------------------
# Single Prediction
# -----------------------------
st.header("📨 Single Message Prediction")

message = st.text_area(
    "Enter Message",
    height=150,
    placeholder="Type your SMS message here..."
)

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        cleaned = clean_text(message)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("🚨 Spam Message")

            st.metric(
                "Spam Confidence",
                f"{probability[1]*100:.2f}%"
            )

        else:

            st.success("✅ Not Spam")

            st.metric(
                "Not Spam Confidence",
                f"{probability[0]*100:.2f}%"
            )

st.divider()


# Batch Prediction

st.header("📂 Batch Spam Detection")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        batch_df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        batch_df = pd.read_csv(uploaded_file, encoding="latin-1")

    batch_df.columns = batch_df.columns.str.strip()

    # If the uploaded file is the original spam dataset
    if "v2" in batch_df.columns:
        batch_df = batch_df.rename(columns={"v2": "message"})

    if "message" not in batch_df.columns:
        st.error("CSV must contain a column named 'message'.")
        st.write("Columns found:", list(batch_df.columns))

    else:
        batch_df["cleaned"] = batch_df["message"].astype(str).apply(clean_text)

        vectors = vectorizer.transform(batch_df["cleaned"])

        predictions = model.predict(vectors)

        probabilities = model.predict_proba(vectors)

        batch_df["Prediction"] = [
            "Spam" if p == 1 else "Not Spam"
            for p in predictions
        ]

        batch_df["Confidence (%)"] = [
            round(max(prob) * 100, 2)
            for prob in probabilities
        ]

        st.success("Prediction Completed Successfully!")

        st.dataframe(
            batch_df[["message", "Prediction", "Confidence (%)"]]
        )

        csv = batch_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="spam_predictions.csv",
            mime="text/csv"
        )