from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from pinecone import Pinecone
from pydantic import BaseModel
import os

load_dotenv()

origins = ["http://localhost:8080", "http://127.0.0.1:8080"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
chat = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    model='gpt-3.5-turbo'
)

embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
indexUpsert = pc.Index('traveltalk-test')
text_field = 'text'
vectorStore = PineconeVectorStore(
    indexUpsert,
    text_key=text_field,
    embedding=embeddings,
)

messages = []

def augmented_prompt(query: str):
    results = vectorStore.similarity_search(query,k=3)
    source_knowledge = '\n'.join([x.page_content for x in results])

    augmented_prompt = f"""  You are a helpful and friendly travel assistant.
    Using 10th grade knowledge, you will answer user queries about anything related to traveling. Questions
    about cities, navigation, cuisine etc.
    If a question arises about a different area, respectfully decline the query.
    Using only the knowledge from the contexts below, answer the query.
    Here is the conversation history: {messages}
    Contexts: {source_knowledge}

    Query: {query}"""
    return augmented_prompt



@app.post('/')
async def create_query(query: Query):
    query_text = query.query
    prompt = HumanMessage(
        content=augmented_prompt(query_text)
    )
    messages.append(prompt)
    res = chat(messages)
    return {res.content}

@app.delete('/')
async def delete():
    global messages
    messages = []
