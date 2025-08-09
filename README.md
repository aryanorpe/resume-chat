---
title: Resume Chat
emoji: 📄
colorFrom: purple
colorTo: green
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: GenAI assistant for HR professionals to do resume analysis.
---

# 📄 Resume Chat

**Resume Chat** is a GenAI-powered assistant for HR recruiters to quickly assess candidate resumes/CVs.  
It uses [Groq's Llama-3.3 70B Versatile](https://groq.com) model to provide conversational analysis of resumes, helping recruiters save time and make better decisions.

---

## 🚀 Features
- **Resume Upload**: Upload PDF, DOCX, or text resumes.
- **Interactive Chat**: Ask questions about the candidate’s resume in plain English.
- **Streaming AI Responses**: Get responses in real-time using Groq's API.
- **Memory Support**: Keeps conversation history for context-aware answers.
- **Custom Model Selection**: Default is `llama-3.3-70b-versatile`.

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aryanorpe/resume-chat.git
   cd resume-chat
   ```

2. **Create a virtual environment**
   ```bash
    python -m venv venv
    source venv/bin/activate   # macOS/Linux
    venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
    pip install -r requirements.txt
   ```

4. **Set your API key**

   - Get your API key from [Groq Console](https://console.groq.com).
   - Create a .env file in the root directory:
   
   ```env
    GROQ_API_KEY=your_api_key_here
   ```

---

## 🖥 Usage

Run the Streamlit app:

   ```bash
    streamlit run app.py
   ```

Once running, open your browser and go to:

   ```bash
    http://localhost:8501
   ```

---

## 📁 Project Structure

   ```bash
    resume-chat/
    │
    ├── app.py                # Main Streamlit app
    ├── styles.css            # Custom UI styling
    ├── requirements.txt      # Python dependencies
    ├── README.md             # Documentation
    └── .env                  # API keys (not committed to Git)
   ```

---

## 📷 Screenshots & 🎥 Demo

Screenshot

Demo Video

(Replace the above image paths and video link with your actual files/links)

---

## ⚙️ Configuration

- Default Model: `llama-3.3-70b-versatile`

- Streaming: Enabled by default

- Session Memory: Maintained using `st.session_state`

---

## 🛡 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Aryan Orpe**

GitHub: [@aryanorpe](https://github.com/aryanorpe)
