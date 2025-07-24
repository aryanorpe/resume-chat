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

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

groq_api_key = os.getenv("GROQ_API_KEY")
# print("groq key", groq_api_key)

client = Groq(
    api_key=groq_api_key,
)

embeddings = HuggingFaceEmbeddings(model_kwargs={"device": "cpu"})
CHROMA_DIR = "chroma_store"
vectorstore = Chroma(embedding_function=embeddings)


def stream_response(stream):
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


# Setting page config
# st.set_page_config(layout="wide")
st.set_page_config(
    page_title="Resume Chat",
    page_icon="📑",
)

# Importing stylesheet
st.markdown(
    "<style>" + open("./styles.css").read() + "</style>", unsafe_allow_html=True
)

st.title("📑💬 Resume Chat")

# Centered GitHub button-style link
st.markdown(
    """
    <div style="text-align: center; margin-top: 10px;">
        <a href="https://github.com/aryanorpe/resume-chat" target="_blank" style="
            background-color: #24292e;
            color: #ffffff;
            padding: 8px 16px;
            text-decoration: none;
            border: 2px solid #ffffff;
            border-radius: 6px;
            font-weight: bold;
            display: inline-block;
            font-family: sans-serif;
            transition: background-color 0.3s, color 0.3s;
        ">⭐ View on GitHub</a>
    </div>
    <br></br>
    """,
    unsafe_allow_html=True,
)

with st.expander(label="What is this?", icon="🤔"):
    st.write(
        "Resume Chat is a GenAI powered assistant for HR Recruiters to quickly assess candidate resumes / CVs."
    )

uploaded_file = st.file_uploader("Upload Resume")


def load_document(uploaded_file):
    file_name = uploaded_file.name
    file_ext = file_name.split(".")[-1].lower()

    # Save to a temp path
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file_name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Choose loader based on file extension
    if file_ext == "txt":
        loader = TextLoader(temp_path, encoding="utf-8")
    elif file_ext == "docx":
        loader = Docx2txtLoader(temp_path)
    elif file_ext == "pdf":
        loader = PyMuPDFLoader(temp_path)  # You can swap with PDFMinerLoader if needed
    else:
        st.error("Unsupported file type: " + file_ext)
        return []

    return loader.load()


# Save uploaded file to a properly closed temp file
if uploaded_file:

    # Load the document
    docs = load_document(uploaded_file)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=0, separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    vectorstore.add_documents(chunks)
    st.success("✅ Resume embedded and stored.")


# Enables streaming responses
def stream_response(stream):
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = "llama-3.3-70b-versatile"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context from vector DB
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