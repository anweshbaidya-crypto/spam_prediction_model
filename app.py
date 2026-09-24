import streamlit as st
import pickle


# Load trained model
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Streamlit interface
st.title("📧 SMS Spam Detector")

st.write("Enter a message below to check whether it is Spam or Not Spam.")


# Text box
message = st.text_area("Enter your message:")


# Prediction button
if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        # Convert message into numerical features
        message_vectorized = vectorizer.transform([message])

        # Make prediction
        prediction = model.predict(message_vectorized)

        # Get probability
        probability = model.predict_proba(message_vectorized)

        if prediction[0] == 1:
            st.error("🚨 This message is SPAM!")

            spam_probability = probability[0][1] * 100

            st.write(
                f"Spam Probability: {spam_probability:.2f}%"
            )

        else:
            st.success("✅ This message is NOT SPAM!")

            spam_probability = probability[0][1] * 100

            st.write(
                f"Spam Probability: {spam_probability:.2f}%"
            )
