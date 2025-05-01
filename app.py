from flask import Flask, render_template, jsonify, request
from src.helper import downlaod_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import ChatMistralAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv(".env", override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY


embeddings = downlaod_hugging_face_embedding()

index_name = "medibot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatMistralAI(temperature=0.4, max_tokens=500)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({'input': msg}) # type: ignore
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)