from pymongo import MongoClient
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain_cohere import ChatCohere
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import os
from dotenv import load_dotenv
import getpass
import os

os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
dbName = "projects"
collectionName = "rag vector search"
collection = client[dbName][collectionName]

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
vectorStore = MongoDBAtlasVectorSearch(collection, embeddings)

def query_data(query):
    docs = vectorStore.similarity_search(query, K=1)
    print(docs)
    as_output = docs[0].page_content

    llm = ChatCohere(model="command", max_tokens=256, temperature=0.1)
    retriever = vectorStore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    retriever_output = qa.invoke(query)
    final_output = retriever_output['result']
    return as_output, final_output

with gr.Blocks(theme=Base(), title="Question-Answering App using Vector Search with RAG") as demo:
    gr.Markdown(
        """
            # Question-Answering App using MongoDB Atlas Vector Search + RAG Architecture
        """
    )
    textbox = gr.Textbox(label="Enter your Question : ", lines=3)
    button = gr.Button("Submit", variant="primary")
    output1 = gr.Textbox(lines=3, max_lines=10, label="Output with just Atlas Vector Search (returns text field as it is) : ")
    output2 = gr.Textbox(lines=3, max_lines=10, label="Output generated by chaining Atlas Vector Search to Langchain's RetrievalQa + Cohere LLM : ")

    button.click(query_data, textbox, outputs=[output1, output2])

demo.launch()
