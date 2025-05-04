import streamlit as st

st.set_page_config(layout="wide")

# Importing stylesheet
st.markdown('<style>' + open('./styles.css').read() + '</style>', unsafe_allow_html=True)

st.title('Resume Chat')

with st.expander(label='What is this?', icon='🤔'):
    st.write('Resume Chat is a GenAI powered assistant for HR Recruiters to quickly assess candidate resumes / CVs.')
    
st.chat_input('hi')