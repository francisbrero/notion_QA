import os
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from dotenv import find_dotenv, load_dotenv
from langchain.agents import initialize_agent
from langchain.agents import Tool
import pinecone


def init_rag(index_name):
    load_dotenv(find_dotenv())

    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    # find your environment next to the api key in pinecone console
    pinecone_env = os.getenv("PINECONE_ENV")
    # get our OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # get the vector database
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    embeddings = OpenAIEmbeddings()
    vectordb = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

    return openai_api_key, vectordb



def create_qa_chain():
    index_name = 'notion-db-chatbot'
    openai_api_key, vectordb = init_rag(index_name)
    # chat completion llm
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )
    # conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )
    # retrieval qa chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )
    # retrieval qa chain with source
    qa_chain_with_source = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        memory=conversational_memory
    )

    return qa_chain, qa_chain_with_source


def query_RAG():
    index_name = 'notion-db-chatbot'
    openai_api_key, vectordb = init_rag(index_name)
    qa_chain, qa_chain_with_source = create_qa_chain(openai_api_key, vectordb)

    return qa_chain, qa_chain_with_source

# main function
if __name__ == "__main__":
    query_RAG()