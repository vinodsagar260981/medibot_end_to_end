from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader 

from langchain_huggingface import HuggingFaceEmbeddings

#Extract PDF from PDF file
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

#Split the DATA into text Chunks perform chunk
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunk = text_splitter.split_documents(extracted_data)
    return text_chunk

#Download the Embeddings from Hugging face
def downlaod_hugging_face_embedding():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings