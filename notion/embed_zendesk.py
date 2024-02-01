import os
from dotenv import load_dotenv
import pinecone
import openai
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import tqdm

# Get variables from .env file
load_dotenv()
openai.api_key = os.getenv('OPEN_AI_API_KEY')
# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone_api_key = os.getenv("PINECONE_API_KEY")
# find your environment next to the api key in pinecone console
pinecone_env = os.getenv("PINECONE_ENV")

index_name = 'notion-db-chatbot'

def load_content_dataframe():
    df = pd.read_csv("./zendesk_data/contents.csv")
    df = df.set_index(["id"])
    print(f"{len(df)} rows in the data. Each row is an article that will be embedded.")
    sample = df.sample(5)
    print("Here is a sample (5 rows)", sample)
    return df

# Function that takes a row of a dataframe, a pinecone index and upserts the embedding to the index
def upsert_embedding(row, index_name):
    # Get the embedding for the row
    embeddings = OpenAIEmbeddings()

    print(row)

    # split the text into chunks of 500 characters with 0 overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(row['content'])

    vectordb = Pinecone.from_documents(all_splits[:1], embeddings, index_name=index_name)

    for i in tqdm.tqdm(range(1, len(all_splits))):
        vectordb.add_documents(all_splits[:i])
    
    return

# Function to upsert the embeddings to Pinecone
def embed_dataframe(df: pd.DataFrame):

    print(f"Generating Pinecone embeddings for index:{index_name}...")
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    
    # upsert each row of the dataframe into the pinecone index, add tqdm for progress bar
    print(f"Upserting Zendesk embeddings to Pinecone index:{index_name}...")
    for row in tqdm.tqdm(df.iterrows()):
        upsert_embedding(row, index_name)

    print(f"Upserted {len(df)} Zendesk embeddings to Pinecone index:{index_name}.")


def main():
    content_df = load_content_dataframe()

    embed_dataframe(content_df)

    print('All done!')
# Entry point
if __name__ == "__main__":
    main()
