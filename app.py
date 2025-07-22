from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_community.llms import ollama
import os
from dotenv import load_dotenv



# Load environment variables from .env file
load_dotenv()      


# Set LangChain API key
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") 
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # Enable LangChain tracing
os.environ['LANGCHAIN_PROJECT'] = "Q&A chatbot with openai" 

prompt= ChatPromptTemplate.from_messages(
    [   
        ('system', 'You are a helpful assistant.respond to the user questions'),                        
        ('user', '{question}')
    ]
)

def generate_response(question, model):
    llm= ollama.Ollama(model=model)
    output_parser = StrOutputParser()   
    chain = prompt | llm | output_parser   
    response = chain.invoke({"question": question})
    return response

# Streamlit app
st.title("Q&A Chatbot")     
st.write("Ask me anything and I will try to answer your questions using Ollama's language model.") 

# Sidebar for settings
st.sidebar.title("Settings")

#dropdown for model selection
model = st.sidebar.selectbox("Select Model", ["llama3:latest", "gemma:2b"])
 

# Input field for user question
user_input = st.text_input("Enter your question here:")

if user_input:
    if model and user_input:
        response = generate_response(user_input, model)
        st.write("Response:", response)
    else:
        st.error("Please enter a question and select a model.")