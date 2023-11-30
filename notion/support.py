# Import necessary modules
import os
from dotenv import find_dotenv, load_dotenv
import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent, initialize_agent
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms.openai import OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from train import init_pinecone_index
from RAG import init_rag
from utils import add_sidebar

# Set up the Streamlit app
st.set_page_config(page_title="🤖 MadKudu: Support Rubook Chat 🧠", page_icon=":robot_face:")
st.title("🤖 MadKudu: Chat with our Notion Support Runbooks 🧠")

# Set up the sidebar
add_sidebar(st)

# initialize the variables
with st.spinner("Initializing..."):
        # get the index
        index_name = 'notion-db-chatbot'
        openai_api_key, vectordb = init_rag(index_name)
st.success("Ready to go!.", icon="✅")

# initialize the LLM
llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages= True
    )

# create the function that retrieves source information from the retriever
def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=retriever
    )
    result = qa_chain.run({'question': query, 'chat_history': st.session_state.messages})
    st.session_state.messages.append((query, result))
    return result

retriever = vectordb.as_retriever()
st.session_state.retriever = retriever

if "messages" not in st.session_state:
    st.session_state.messages = []    
#
for message in st.session_state.messages:
    st.chat_message('human').write(message[0])
    st.chat_message('ai').write(message[1])    
#
if query := st.chat_input():
    st.chat_message("human").write(query)
    response = query_llm(st.session_state.retriever, query)
    st.chat_message("ai").write(response)