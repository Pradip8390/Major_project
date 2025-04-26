import os
import time
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from deep_translator import GoogleTranslator
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai

st.set_page_config(page_title="NyayaBot", layout="wide")
st.header("AI Chatbot for Department of Justice Website")

with st.sidebar:
    st.title("NyayaBot")
    col1, col2, col3 = st.columns([1, 30, 1])
    with col2:
        st.image("images/Judge.png", use_container_width=True)
    selected_language = st.selectbox("Start by Selecting your Language", 
                                     ["English", "Assamese", "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", 
                                      "Nepali", "Odia", "Punjabi", "Sindhi", "Tamil", "Telugu", "Urdu"])

genai.configure(api_key="AIzaSyADVzPdi02iVFomiXifFQhLZiyOMC0NV84")
model = genai.GenerativeModel('gemini-1.5-pro') 

def hide_hamburger_menu():
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

hide_hamburger_menu()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

@st.cache_resource
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
    return FAISS.from_texts(text_chunks, embedding=embeddings)

@st.cache_resource
def load_legal_data():
    text_data = """
    [Your legal text data here]
    """
    chunks = get_text_chunks(text_data)
    return get_vector_store(chunks)

vector_store = load_legal_data()
db_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def get_response(prompt, context):
    full_prompt = f"""
    As a legal chatbot specializing in the Indian Penal Code and Department of Justice services, provide precise legal responses.
    
    CONTEXT: {context}
    QUESTION: {prompt}
    ANSWER:
    """
    return model.generate_content(full_prompt, stream=True)

def translate_answer(answer, target_language):
    translator = GoogleTranslator(source='auto', target=target_language)
    return translator.translate(answer)

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

def get_trimmed_chat_history():
    max_history = 10
    return st.session_state.messages[-max_history:]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

input_prompt = st.chat_input("Start with your legal query")
if input_prompt:
    st.session_state.messages.append({"role": "user", "content": input_prompt})
    
    with st.chat_message("user"):
        st.markdown(input_prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = "⚠️ **_Gentle reminder: We generally ensure precise information, but do double-check._** \n\n"
        
        context = db_retriever.get_relevant_documents(input_prompt)
        context_text = "\n".join([doc.page_content for doc in context])
        
        response_stream = get_response(input_prompt, context_text)
        for chunk in response_stream:
            full_response += chunk.text
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.01)
        
        if selected_language != "English":
            translated_response = translate_answer(full_response, selected_language.lower())
            message_placeholder.markdown(translated_response)
        else:
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    if st.button('🗑️ Reset Conversation'):
        reset_conversation()
        st.experimental_rerun()

def footer():
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: black;
            text-align: center;
        }
        </style>
        <div class="footer">
        </div>
        """, unsafe_allow_html=True)

footer()
