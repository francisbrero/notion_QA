import os
from dotenv import find_dotenv, load_dotenv
from langchain.vectorstores import Pinecone
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings
import pinecone
import time


# Load environment variables from .env file
load_dotenv(find_dotenv())

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone_api_key = os.getenv("PINECONE_API_KEY")
# find your environment next to the api key in pinecone console
pinecone_env = os.getenv("PINECONE_ENV")

LLaMa_model_path="../../../llama/llama-2-7b-chat/ggml-model-f16_q4_0.gguf"
NOTION_DIRECTORY_PATH="../notion_data/Support_runbook"

loader = NotionDirectoryLoader(NOTION_DIRECTORY_PATH)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

embeddings = LlamaCppEmbeddings(model_path=LLaMa_model_path)

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

index_name = 'notion-db-chatbot'

if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

# we create a new index
pinecone.create_index(
    name=index_name,
    metric='dotproduct',
    dimension=4096  # 4096 is the dimensionality of the LLaMa model
)

# wait for index to be initialized
while not pinecone.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pinecone.Index(index_name)

# we add the documents to the index, add tqdm to track progress
vectordb = Pinecone.from_documents(all_splits, embeddings, index_name=index_name)