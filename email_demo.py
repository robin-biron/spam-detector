"""Text-only predictions using the article's fitted Model A components."""

MAX_EMAIL_LENGTH = 20_000

# Fictional messages: no real mailbox content or clickable destinations.
EXAMPLES = {
    "Spam": {
        "Prize offer": "CONGRATULATIONS! You have won a FREE cash prize. Claim your money now! Limited time offer. Reply today to receive your reward.",
        "Sales promotion": "SPECIAL OFFER! Buy now and save money. Get cheap products at the lowest prices. This exclusive deal is available today only. Order now!",
    },
    "Ham": {
        "Python bug report": "I have checked the bug in the Python script. The function returns None because the variable is not initialized. Could you run the tests again and post the traceback?",
        "Lunch invitation": "Hi, are you free for lunch on Friday? We could meet at the usual place around noon. Let me know what works for you. Have a nice week!",
    },
}


def classify_email(text, vectorizer, classifier):
    """Transform with the existing vocabulary; never fit on a submitted email."""
    text = text.strip()
    if not text:
        raise ValueError("Enter an email before analyzing it.")
    if len(text) > MAX_EMAIL_LENGTH:
        raise ValueError(f"Keep the email below {MAX_EMAIL_LENGTH:,} characters.")
    features = vectorizer.transform([text])
    if features.nnz == 0:
        raise ValueError("No words match the model's English vocabulary. Try a longer email in English.")
    probabilities = dict(zip(classifier.classes_, classifier.predict_proba(features)[0]))
    prediction = int(classifier.predict(features)[0])
    return {"label": "Spam" if prediction == 1 else "Ham", "spam_probability": float(probabilities[1])}


def show_prediction(result):
    import streamlit as st

    if result["label"] == "Spam":
        st.warning("Prediction: Spam")
    else:
        st.success("Prediction: Ham (legitimate email)")
    st.metric("Estimated spam probability", f"{result['spam_probability']:.1%}")
    st.caption("This is the model's estimate, not a security guarantee or a calibrated risk score.")


def render_examples(vectorizer, classifier):
    import streamlit as st

    st.header("Try spam and ham examples")
    st.write("Choose a fictional English email, then compare its illustrative label with the model's prediction.")
    category = st.radio("Example type", list(EXAMPLES), horizontal=True)
    title = st.selectbox("Choose an example", list(EXAMPLES[category]))
    text = EXAMPLES[category][title]
    st.caption(f"Illustrative label: {category}")
    st.text_area("Example email", text, height=180, disabled=True)
    if st.button("Analyze example", type="primary"):
        result = classify_email(text, vectorizer, classifier)
        show_prediction(result)
        if result["label"] != category:
            st.info("The prediction differs from the illustrative label. Short examples can expose the model's limitations.")


def render_email_form(vectorizer, classifier):
    import streamlit as st

    st.header("Analyze an email")
    st.write("Paste the body of an email in English. Headers and subject lines are not required.")
    st.caption("The text is processed on the Streamlit server when you submit it. The app does not save submitted emails; use a fictional or anonymized message.")
    with st.form("email_analysis"):
        text = st.text_area("Email body", height=220, max_chars=MAX_EMAIL_LENGTH,
                            placeholder="Paste an English email here...")
        submitted = st.form_submit_button("Analyze email", type="primary")
    if submitted:
        try:
            result = classify_email(text, vectorizer, classifier)
        except ValueError as error:
            st.error(str(error))
        else:
            show_prediction(result)
