import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader  # or PDFMinerLoader
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Setting page config
# st.set_page_config(layout="wide")
st.set_page_config(page_title='Resume Chat',
                   page_icon='📑',)

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def stream_response(stream):
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


# Importing stylesheet
st.markdown('<style>' + open('./styles.css').read() + '</style>', unsafe_allow_html=True)

st.title('Resume Chat')
st.header('Project by Aryan Orpe')

with st.expander(label='What is this?', icon='🤔'):
    st.write('Resume Chat is a GenAI powered assistant for HR Recruiters to quickly assess candidate resumes / CVs.')
    
st.file_uploader('Upload Resume')

if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = "meta-llama/llama-4-maverick-17b-128e-instruct"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["groq_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream_response(stream))
    st.session_state.messages.append({"role": "assistant", "content": response})

# st.chat_input('hi')