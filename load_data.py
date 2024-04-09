from pymongo import MongoClient
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
# from lan
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
dbName = "projects"
collectionName = "rag vector search"
collection = client[dbName][collectionName]

loader = DirectoryLoader('./sample_files', glob='./*.txt', show_progress=True)
data = loader.load()

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

vectorStore = MongoDBAtlasVectorSearch.from_documents(data, embeddings, collection=collection)
