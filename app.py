import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load Kaggle Mail Data
@st.cache_data
def load_data():
    df = pd.read_csv("mail_data.csv")
    df = df.rename(columns={"Category": "label", "Message": "message"})  # adjust if needed
    return df

# Train and evaluate model
@st.cache_resource
def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)
    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('nb', MultinomialNB())
    ])
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, accuracy, report

# --- Streamlit UI ---
st.set_page_config(page_title="Spam Detector", page_icon="📧")
st.title("📧 Email Spam Detection App")
st.markdown("Using Kaggle Mail Data — Built with **Naive Bayes** and **Streamlit**")

# Load and train
data = load_data()
model, accuracy, report = train_model(data)

# 📊 Label Distribution
st.subheader("📊 Dataset Distribution")
label_counts = data['label'].value_counts()

fig, ax = plt.subplots()
ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=["#66b3ff", "#ff9999"])
ax.axis('equal')
st.pyplot(fig)

# ✅ Accuracy
st.subheader("✅ Model Performance")
st.write(f"**Accuracy:** {accuracy*100:.2f}%")

# Optional Report
with st.expander("📋 View Detailed Classification Report"):
    st.dataframe(pd.DataFrame(report).transpose())

# 📝 Input Section
st.subheader("✉️ Enter Email Message")
input_msg = st.text_area("Paste email content here:")

if st.button("Detect"):
    if input_msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([input_msg])[0]
        if prediction.lower() == "spam":
            st.error("🚫 This message is likely **SPAM**")
        else:
            st.success("✅ This message is **HAM (Not Spam)**")

# 📂 View Dataset
with st.expander("🔍 View Sample Dataset"):
    st.dataframe(data.sample(10))
