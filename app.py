import os
from langchain.chat_models import ChatOpenAI
import openai
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from operator import itemgetter
from dotenv import load_dotenv
import bs4 
from bs4 import SoupStrainer
#from langchain_openai import OpenAIEmbeddings
load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")#Document loader
#print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
#PINECONE_API_KEY=os.getenv('PINECONE_API_KEY ')
@st.cache_data
def load_document_loader():
    loader = WebBaseLoader(
    'https://en.wikipedia.org/wiki/Elon_Musk',
    bs_kwargs=dict(parse_only=SoupStrainer(class_=('mw-content-ltr mw-parser-output')))
    )
    documents = loader.load()
    # Split documents into chunks
    recursive = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = recursive.split_documents(documents)
    return chunks 
chunks=load_document_loader()
# Initialize embedding and Qdrant
embed = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     openai_api_key=openai.api_key,
#     # With the `text-embedding-3` class
#     # of models, you can specify the size
#     # of the embeddings you want returned.
#     # dimensions=1024
# )
    # of the embeddings you want returned.
    # dimensions=1024
# Qdrant setup
api_key = os.getenv('qdrant_api_key')
url = 'https://1328bf7c-9693-4c14-a04c-f342030f3b52.us-east4-0.gcp.cloud.qdrant.io:6333'
doc_store = QdrantVectorStore.from_existing_collection(
    embedding=embed,
    url=url,
    api_key=api_key,
    prefer_grpc=True,
    collection_name="Elon Muske"
)
# pineconedb=PineconeVectorStore.from_existing_index(index_name='project1', embedding=embeddings)
# LLM = ChatOpenAI(
#                 model_name='gpt-4o-mini',
#                 openai_api_key=openai.api_key,
#                 temperature=0)
# Initialize Google LLM
google_api = os.getenv('google_api_key')
llm = GoogleGenerativeAI(model="gemini-1.5-flash-002", google_api_key=google_api)

# Setup retriever and chain

num_chunks = 5
#retriever = pineconedb.as_retriever(search_type="mmr", search_kwargs={"k": num_chunks})
retriever =doc_store.as_retriever(search_type="mmr", search_kwargs={"k": num_chunks})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt_str = """
You are a highly knowledgeable and conversational chatbot specializing in providing accurate and insightful information about Elon Musk.
Answer all questions as if you are an expert on his life, career, companies, and achievements.
Context: {context}
Question: {question}
"""
_prompt = ChatPromptTemplate.from_template(prompt_str)

# Chain setup
query_fetcher = itemgetter("question")
setup = {"question": query_fetcher, "context": query_fetcher | retriever| format_docs }
_chain = setup | _prompt | llm | StrOutputParser()

# Streamlit UI
# Streamlit UI
st.title("Ask Anything About Elon Musk")

# Chat container to display conversation
chat_container = st.container()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def send_input():
    st.session_state.send_input=True
# Input field for queries
with st.container():
    query = st.text_input("Please enter a query", key="query", on_change=send_input)
    send_button = st.button("Send", key="send_btn")  # Single send button

# Chat logic
if send_button or send_input and query:
    with st.spinner("Processing... Please wait!"):  # Spinner starts here
        response = _chain.invoke({'question': query})  # Generate response
    # Update session state with user query and AI response
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("ai", response))

with chat_container:
    for role, message in st.session_state.messages:
        st.chat_message(role).write(message)
