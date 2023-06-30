"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pathlib import Path


def ingest_docs():
    """Get documents from web pages."""
    #loader = ReadTheDocsLoader(Path("./A.html"))
    #raw_documents = loader.load()
    #text_splitter = RecursiveCharacterTextSplitter(
    #    chunk_size=1000,
    #    chunk_overlap=200,
    #)
    #documents = text_splitter.split_documents(raw_documents)
    
    with open('files/churchill_speech.txt') as f:
        churchill_speech = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len
    )
    
    documents = text_splitter.create_documents([churchill_speech])
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
