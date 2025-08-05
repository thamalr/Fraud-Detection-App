import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
import time
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

# --- Preprocessing Utilities ---
def cat_num_features(df):
    catf = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5',
            'card6', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
            'M4', 'DeviceType', 'DeviceInfo', 'TransactionWD',
            'P_parent_domain', 'P_domain_name', 'P_top_level_domain',
            'R_parent_domain', 'R_domain_name', 'R_top_level_domain',
            'device_name', 'device_version']
    catf += ['id_' + str(i) for i in range(12, 39)]
    catf = [f for f in catf if f in df.columns]
    numf = [f for f in df.columns if f not in catf and f != 'isFraud']
    return catf, numf

def label_encode(df, catf):
    for f in catf:
        df[f] = df[f].astype(str)
        le = LabelEncoder()
        df[f] = le.fit_transform(df[f])
    return df

def frequency_encode(df, features):
    for f in features:
        vc = df[f].value_counts(dropna=True, normalize=True).to_dict()
        name = f + '_FE'
        df[name] = df[f].map(lambda x: vc.get(x, -1))
    return df

# --- Load Models ---
import os

# Get base directory (repo root, one level above /app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load models from absolute path
import xgboost as xgb
xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(MODEL_DIR, "xgb_model.json"))

tokenizer = DistilBertTokenizerFast.from_pretrained(os.path.join(MODEL_DIR, "distilbert_finetuned"))
nlp_model = DistilBertForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, "distilbert_finetuned"))

nlp_model.eval()

st.title("🔊 Fraud Detection App")
tab1, tab2 = st.tabs(["Batch Inference", "Text Inference"])

# --- Batch Prediction Tab ---
with tab1:
    st.header("📂 Upload CSV for Batch Fraud Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        if st.button("Run Batch Prediction"):
            with st.spinner("Running batch predictions..."):
                time.sleep(0.5)
                df_input = pd.read_csv(uploaded_file)
                if "TransactionID" not in df_input.columns:
                    st.error("TransactionID column is required in the input file.")
                else:
                    tx_ids = df_input["TransactionID"]
                    df = df_input.drop(columns=["TransactionID"])
                    catf, numf = cat_num_features(df)
                    df = label_encode(df, catf)
                    df = frequency_encode(df, catf)
                    X = df.drop(columns=['isFraud'], errors='ignore')
                    X = X.reindex(columns=xgb_model.feature_names, fill_value=0)
                    y_pred = xgb_model.predict(X)
                    df_output = pd.DataFrame({"TransactionID": tx_ids, "Prediction": y_pred})

                    # Summary section
                    st.subheader("📈 Summary")
                    total_tx = len(df_output)
                    fraud_count = (df_output["Prediction"] == 1).sum()
                    non_fraud_count = (df_output["Prediction"] == 0).sum()

                    st.markdown(f"- Total Transactions: **{total_tx}**")
                    st.markdown(f"- Fraudulent Transactions: **{fraud_count}**")
                    st.markdown(f"- Legitimate Transactions: **{non_fraud_count}**")

                    # Pie chart
                    counts = df_output["Prediction"].value_counts().sort_index()
                    labels = ["Not Fraud", "Fraud"]
                    actual_labels = [labels[i] for i in counts.index]

                    fig, ax = plt.subplots()
                    ax.pie(counts, labels=actual_labels, autopct='%1.1f%%')
                    st.pyplot(fig)

                    st.subheader("Sample Output")
                    st.dataframe(df_output.head(10))

                    csv = df_output.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Predictions", csv, "fraud_predictions.csv", "text/csv")

# --- NLP Model Tab ---
with tab2:
    st.header("📝 Enter a Transaction Description")
    text = st.text_area("Description", "")
    if st.button("Predict Text Fraud"):
        if text.strip():
            with st.spinner("Classifying the transaction description..."):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = nlp_model(**inputs)
                    probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
                    prediction = torch.argmax(outputs.logits, dim=1).item()
                st.subheader("Prediction Result")
                st.success("Fraud" if prediction == 1 else "Not Fraud")
                st.markdown(f"**Confidence:** {probs[prediction]*100:.2f}%")
                st.progress(probs[prediction])
        else:
            st.warning("Please enter a description before clicking Predict.")
