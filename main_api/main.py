from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from pinecone import Pinecone
import os
import json
from pydantic import BaseModel
load_dotenv()

class Query(BaseModel):
    query: str

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
chat = ChatOpenAI(
    openai_api_key = os.environ['OPENAI_API_KEY'],
    model = 'gpt-3.5-turbo'
)

embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

index_name = 'traveltalk'
indexUpsert = pc.Index('traveltalk')
text_field = 'text'
vectorStore = PineconeVectorStore(
        indexUpsert,
        text_key=text_field,
        embedding=embeddings,
    )

messages = [
    SystemMessage(content='You are a helpfull and friendly travel assistant.')
]

# query ='whats the phone number of Parish of the Blessed Trinity in amsterdam?'


def augmented_prompt(query: str):
    results = vectorStore.similarity_search(query,k=3)
    source_knowledge = '\n'.join([x.page_content for x in results])

    augmented_prompt = f"""Using your learned knowledge answer the query, If you dont have that knowledge, answer the query using the context below

    Contexts: {source_knowledge}

    Query: {query}"""
    return augmented_prompt

app = FastAPI()

@app.post('/')

async def create_query(query: Query):
    # Extract the query text from the Query object
    query_text = query.query

    # Convert the query text to raw JSON string
    raw_query_text = json.dumps({"query": query_text})

    # Pass the raw query text to augmented_prompt function
    prompt = HumanMessage(
        content=augmented_prompt(raw_query_text)
    )
    messages.append(prompt)
    res = chat(messages)
    return {res.content}