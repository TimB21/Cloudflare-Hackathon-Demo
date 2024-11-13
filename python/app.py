import json
import requests
import streamlit as st
from cloudflare import Cloudflare
from PIL import Image
import io

# Set Hugging Face API URL and headers
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
# Set Hugging Face API URL and headers using the token from Streamlit secrets
hugging_token = st.secrets["HUGGINGFACE_API_TOKEN"]
headers = {"Authorization": f"Bearer {hugging_token}"}  # Correct way to reference the token

def query(payload):
    """Function to query the Hugging Face API to generate an image."""
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

# Streamlit UI setup
st.title("Ghost Writer AI")
st.subheader("Generate custom song lyrics based on your favorite artist and theme")

# Set Cloudflare API key from Streamlit secrets
client = Cloudflare(api_token=st.secrets["CLOUDFLARE_API_TOKEN"])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input for artist and theme
artist = st.text_input("Enter the musical artist:")
theme = st.text_input("Enter a theme or description for the song:")

# Generate song lyrics when both artist and theme are provided
if artist and theme:
    # Construct prompt for AI model
    prompt = f"Write a song in the style of {artist} about {theme}."
    
    # Add prompt to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user prompt in chat message container
    with st.chat_message("user"):
        st.markdown(f"**Artist:** {artist}\n**Theme:** {theme}")
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with client.workers.ai.with_streaming_response.run(
            account_id=st.secrets["CLOUDFLARE_ACCOUNT_ID"],
            model_name="@cf/meta/llama-3.1-8b-instruct",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ) as response:
            # Create a token iterator for response
            def iter_tokens(r):
                for line in r.iter_lines():
                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                        entry = json.loads(line.replace("data: ", ""))
                        yield entry["response"]

            completion = st.write_stream(iter_tokens(response))
    st.session_state.messages.append({"role": "assistant", "content": completion})
    
    # After generating lyrics, request Hugging Face API to generate an album cover
    album_cover_prompt = f"Album cover for {artist} with a theme of {theme}, highly artistic and visually captivating."
    album_cover_bytes = query({
        "inputs": album_cover_prompt,
    })
    
    # Display the generated album cover image
    album_cover_image = Image.open(io.BytesIO(album_cover_bytes))
    st.image(album_cover_image, caption=f"Album Cover for {artist} - {theme}", use_column_width=True)
