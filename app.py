import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
from pydub import AudioSegment
import speech_recognition as sr
import os

# Load API key from environment variable (recommended for Hugging Face)
api_key = os.environ.get("GEMINI_API_KEY")

# Define the query function
def query(user_query):
    chat_template = ChatPromptTemplate.from_messages([
        ("system", 
        """You are a professional and personal Male query chatbot for Tamanna.
You give friendly and short replies to greeting-type queries like "hi", "hello", or "how are you".
Whenever someone asks about your identity, clearly state that you are the personal query chatbot of Tamanna.
You always provide accurate, simple, and helpful answers to any kind of question from the user.
Whenever user asks 'What you do ?', just tell them you solve queries.
If user asks you to use another language while chatting, switch to that language.
if user asks your name say 🤖 Auralis ( Auree for Tamanna ).'"""
        ),
        ("human", 
        "I may have a low IQ and many questions. Please chat with me in a kind and simple way. Here's my first question: {query}"
        )
    ])

    parser = StrOutputParser()

    model = ChatGoogleGenerativeAI(model='models/gemini-1.5-flash-latest', google_api_key=api_key)
    chain = chat_template | model | parser
    result = chain.invoke({'query': user_query})
    return result

# Define the transcription function
def transcribe(audio_path):
    recognizer = sr.Recognizer()
    wav_path = "converted.wav"
    try:
        AudioSegment.from_mp3(audio_path).export(wav_path, format="wav")
    except Exception as e:
        return f"⚠️ Error converting MP3 to WAV: {e}"

    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        transcription = recognizer.recognize_google(audio_data)
        return transcription
    except sr.UnknownValueError:
        return "❌ Could not understand the audio."
    except sr.RequestError as e:
        return f"⚠️ API request error: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"

# Combined chatbot function
def chatbot(text_input, audio_input):
    if text_input:
        user_query = text_input
    elif audio_input:
        user_query = transcribe(audio_input)
    else:
        return "❌ Please enter text or record audio."

    bot_response = query(user_query)
    return bot_response

# Gradio UI
with gr.Blocks(theme=gr.themes.Base()) as app:
    gr.Markdown("## 🤖 Auralis : Personal Chatbot for Tamanna .")

    chatbot_output = gr.Chatbot(label="Conversation")

    with gr.Row():
        text_input = gr.Textbox(placeholder="Ask anything...", show_label=False, container=False)
        audio_input = gr.Audio(sources="microphone", type="filepath", label=None)
        submit_btn = gr.Button("Send")

    def process_inputs(text, audio, history):
        response = chatbot(text, audio)
        user_message = text if text else "🎤 (Voice Input)"
        history = history or []
        history.append((user_message, response))
        return history, None, None

    submit_btn.click(
        process_inputs,
        inputs=[text_input, audio_input, chatbot_output],
        outputs=[chatbot_output, text_input, audio_input]
    )

    text_input.submit(
        process_inputs,
        inputs=[text_input, audio_input, chatbot_output],
        outputs=[chatbot_output, text_input, audio_input]
    )

# Launch the app
app.launch()