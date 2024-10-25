import streamlit as st
import requests

# FastAPI URL for the ask endpoint
API_URL = "http://localhost:8093/ask"

st.set_page_config(
    page_title="Spotify Review Chatbot",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .score-box {
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    div[data-testid="stVerticalBlock"] > div:nth-child(2) {
        display: flex;
        align-items: center;
        justify-content: flex-end;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎵 Spotify Review Chatbot")
st.write("Ask any question about Spotify reviews, and the chatbot will respond based on the available reviews.")

# Create two columns for the input area
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input("Your question:", placeholder="Type your question here...")

with col2:
    ask_button = st.button("Ask", type="primary")


if ask_button:
    if user_input:
        try:
            # Show a spinner while waiting for the response
            with st.spinner("Getting response..."):
                headers = {
                    "content-type": "application/json",
                    "x-api-key": "ebce2698dadf0593c979a2798c84e49a0"
                }
                response = requests.post(
                    API_URL,
                    json={"user_input": user_input},
                    headers=headers,
                    timeout=30  # Add timeout to prevent hanging
                )

            if response.status_code == 200:
                result = response.json()
                chatbot_response = result.get("response", "No response received.")
                score = result.get("score", 0.0)

                # Display the response in a container
                with st.container():
                    st.markdown("### 💬 Chatbot's Response")
                    st.write(chatbot_response)

                    # Display the score with custom styling
                    st.markdown("### 📊 Response Score")
                    score_percentage = score * 100

                    # Create a color gradient from red to green based on the score
                    color = f"rgb({int(255 * (1 - score))}, {int(255 * score)}, 0)"

                    st.markdown(
                        f"""
                        <div class="score-box">
                            <p style="color: {color}; font-size: 24px; font-weight: bold; margin: 0;">
                                {score_percentage:.1f}%
                            </p>
                            <p style="color: gray; font-size: 14px; margin: 0;">
                                of reviews represented in the response
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:
                st.error(f"Error: {response.status_code}")
                st.write("Error details:", response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the server: {str(e)}")

    else:
        st.warning("⚠️ Please enter a question.")

# Add a footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        Built with Streamlit • Powered by LangGraph and GPT4o
    </div>
    """,
    unsafe_allow_html=True
)
