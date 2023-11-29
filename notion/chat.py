import os
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from dotenv import find_dotenv, load_dotenv
from langchain.agents import initialize_agent
from langchain.agents import Tool
import pinecone


def init(index_name):
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



def create_agent(openai_api_key, vectordb):
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
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )

    tools = [
        Tool(
            name='Knowledge Base',
            func=qa.run,
            description=(
                'use this tool when answering general knowledge queries to get '
                'more information about the topic'
            )
        )
    ]

    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=conversational_memory
    )

    return agent

def get_prompt(agent):
    """Get a prompt from the user and run it through the agent."""
    print("Type 'exit' to quit")

    while True:
        prompt = input("Enter a prompt: ")

        if prompt.lower() == 'exit':
            print('Exiting...')
            break
        else:
            try:                
                # add the question to the conversation chain
                results = agent.run(input=prompt)
                print(results)
            except Exception as e:
                print(e)

def run_agent():
    index_name = 'notion-db-chatbot'
    openai_api_key, vectordb = init(index_name)
    agent = create_agent(openai_api_key, vectordb)
    get_prompt(agent)


# run our main function
if __name__ == '__main__':
    run_agent()