from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from pinecone import Pinecone
import os
load_dotenv()


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
    SystemMessage(content='You are a helpfull and friendly travel assistant. When answering a user message make it clear if that was the data you were trained on, or if it was added context')
]

query ='whats the phone number of Parish of the Blessed Trinity in amsterdam?'


def augmented_prompt(query: str):
    results = vectorStore.similarity_search(query,k=3)
    source_knowledge = '\n'.join([x.page_content for x in results])

    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts: {source_knowledge}

    Query: {query}"""
    return augmented_prompt

app = FastAPI()

@app.post('/')
async def root():
    prompt = HumanMessage(
    content=augmented_prompt(query)
)
    messages.append(prompt)
    res = chat(messages)
    return {res.content}