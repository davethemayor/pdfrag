from flask import Flask, request, jsonify, render_template
import openai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import time
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Define the directory and metadata file paths
pdf_dir = "./pdfs"
metadata_file = "./chroma_db_nccn/last_update.txt"

# Load the last update time
if os.path.exists(metadata_file):
    with open(metadata_file, "r") as f:
        last_update_time = datetime.fromisoformat(f.read().strip())
else:
    last_update_time = datetime.min

# List PDF files that have been modified since the last update
pdf_files = [
    f
    for f in os.listdir(pdf_dir)
    if f.endswith(".pdf")
    and datetime.fromtimestamp(os.path.getmtime(os.path.join(pdf_dir, f)))
    > last_update_time
]
print(
    f"Found {len(pdf_files)} new or modified PDF files in '{pdf_dir}'"
)  # Debugging output

if pdf_files:
    # Load new or modified PDFs
    loaders = [PyPDFLoader(os.path.join(pdf_dir, file)) for file in pdf_files]
    print("PDF loaders created")  # Debugging output

    docs = []
    for loader in loaders:
        loaded_docs = loader.load()
        print(f"Loaded {len(loaded_docs)} documents from {loader.file_path}")
        docs.extend(loaded_docs)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs)
    print(f"Split documents into {len(docs)} chunks")  # Debugging output

    # Define embedding function
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    print("Embedding function defined")  # Debugging output

    # Create a vector store or add documents to an existing one
    vectorstore = Chroma.from_documents(
        docs, embedding_function, persist_directory="./chroma_db_nccn"
    )
    print("Vector store updated with new or modified documents")  # Debugging output

    # Update the last update time
    with open(metadata_file, "w") as f:
        f.write(datetime.now().isoformat())

    # Print the count of documents in the vector store
    print(f"Number of documents in the vector store: {vectorstore._collection.count()}")
else:
    print("No new or modified PDF files to process")

app = Flask(__name__)

openai.api_key = os.environ["OPENAI_API_KEY"]

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_db = Chroma(
    persist_directory="./chroma_db_nccn", embedding_function=embedding_function
)

history = [
    {
        "role": "system",
        "content": 'You are an assistant for question-answering tasks. Use the pieces of retrieved context to answer the questions. If you cannot find the answer in the context, just say "I don\'t have that information"',
    },
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["user_input"]

    # Perform similarity search using vector_db
    search_results = vector_db.similarity_search(user_input, k=3)
    some_context = "\n\n".join([str(result) for result in search_results])

    # Append search context to the user input
    context_user_input = "Query: " + user_input + "\n\n\nContext: " + some_context

    new_message = {"role": "user", "content": context_user_input}
    history.append(new_message)

    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        temperature=0.7,
        stream=False,
    )

    response = completion.choices[0].message.content
    history.append({"role": "assistant", "content": response})

    return jsonify({"response": response, "history": history})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
