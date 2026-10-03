import re
import joblib
import streamlit as st

# Same cleaning as in the notebook (the app must clean text the same way)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

model = joblib.load("emergency_model.joblib")

st.title("Emergency Call Classifier")
st.write("Type what the caller said, and the model will suggest the emergency category.")

call = st.text_area("Caller's words:", placeholder="e.g. There's smoke coming from my kitchen!")

if st.button("Classify"):
    if call.strip() == "":
        st.warning("Please type a call first.")
    else:
        cleaned = clean_text(call)
        prediction = model.predict([cleaned])[0]
        st.success(f"Predicted category: **{prediction}**")

        # Show how confident the model is for each category
        probs = model.predict_proba([cleaned])[0]
        st.write("Confidence per category:")
        st.bar_chart(dict(zip(model.classes_, probs)))
