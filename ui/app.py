import streamlit as st
import requests

# FastAPI URL for the ask endpoint
API_URL = "http://localhost:8093/ask"

st.title("Spotify Review Chatbot")
st.write("Ask any question about Spotify reviews, and the chatbot will respond based on the available reviews.")

# User input for the question
user_input = st.text_input("Your question:", "")

# Button to send the request
if st.button("Ask"):
    if user_input:
        # Make the POST request to FastAPI
        headers = {"content-type": "application/json", "x-api-key": "ebce2698dadf0593c979a2798c84e49a0"}
        response = requests.post(API_URL, json={"user_input": user_input}, headers=headers)

        if response.status_code == 200:
            # Display the chatbot's response
            result = response.json().get("response", "No response received.")
            st.write("### Chatbot's Response:")
            st.write(result)
        else:
            # Display error if the API call fails
            st.write("Error:", response.status_code, response.text)
    else:
        st.write("Please enter a question.")

