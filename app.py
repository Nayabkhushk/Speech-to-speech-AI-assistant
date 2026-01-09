import streamlit as st
import os
import whisper
import tempfile
from groq import Groq
from gtts import gTTS
from audiorecorder import audiorecorder

st.set_page_config(page_title="Speech AI Assistant")

# Load API key from secrets
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=os.environ["GROQ_API_KEY"])

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

def speech_to_text(path):
    return whisper_model.transcribe(path, language="en")["text"]

def groq_chat(text):
    st.session_state.conversation.append({"role": "user", "content": text})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=st.session_state.conversation
    )

    reply = response.choices[0].message.content
    st.session_state.conversation.append({"role": "assistant", "content": reply})
    return reply

def speak(text):
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    gTTS(text=text, lang="en").save(file.name)
    return file.name

st.title("🎙 Live Speech-to-Speech AI")

audio = audiorecorder("Start Recording", "Stop Recording")

if len(audio) > 0:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio.export(f.name, format="wav")
        user_text = speech_to_text(f.name)

    st.write("🧑 You:", user_text)

    ai_reply = groq_chat(user_text)
    st.write("🤖 AI:", ai_reply)

    st.audio(speak(ai_reply))
