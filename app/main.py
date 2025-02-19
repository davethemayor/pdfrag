from flask import Flask, request, jsonify, render_template
import openai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

openai.api_key = os.environ["OPENAI_API_KEY"]

app = Flask(__name__)

# Define the directory and metadata file paths
pdf_dir = "/pdfs"
metadata_file = "./chroma_db_nccn/last_update.txt"
log_file = "chat.log"

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
print(f"Found {len(pdf_files)} new or modified PDF files in '{pdf_dir}'")

if pdf_files:
    # Load new or modified PDFs
    loaders = [PyPDFLoader(os.path.join(pdf_dir, file)) for file in pdf_files]
    print("PDF loaders created")

    docs = []
    for loader in loaders:
        loaded_docs = loader.load()
        print(f"Loaded {len(loaded_docs)} documents from {loader.file_path}")
        docs.extend(loaded_docs)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs)
    print(f"Split documents into {len(docs)} chunks")

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # Create a vector store or add documents to an existing one
    vectorstore = Chroma.from_documents(
        docs, embedding_function, persist_directory="./chroma_db_nccn"
    )

    # Update the last update time
    with open(metadata_file, "w") as f:
        f.write(datetime.now().isoformat())

    print(f"Number of documents in the vector store: {vectorstore._collection.count()}")
else:
    print("No new or modified PDF files to process")


embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_db = Chroma(
    persist_directory="./chroma_db_nccn", embedding_function=embedding_function
)

SYSTEM_PROMPT = {
    "role": "system",
    "content": 'You are an assistant for question-answering tasks. Use the pieces of retrieved context to answer the questions. Cite sources with hyperlinks to the documents. Append the page number to the hyperlink URL in the format: "/pdfs/doc.pdf#page=n" and use the filename + page number as the display text. If you cannot find the answer in the context, just say "I don\'t have that information"',
}


def log_interaction(user_input, response):
    with open(log_file, "a") as f:
        log_entry = f"{datetime.now().isoformat()} - User: {user_input}\nAssistant: {response}\n\n"
        f.write(log_entry)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["user_input"]
    user_history = request.json.get("history", [])

    # Perform similarity search using vector_db
    search_results = vector_db.similarity_search(user_input, k=3)
    some_context = "\n\n".join([str(result) for result in search_results])

    # Append search context to the user input
    context_user_input = "Query: " + user_input + "\n\n\nContext: " + some_context

    new_message = {"role": "user", "content": context_user_input}
    history = [SYSTEM_PROMPT] + user_history + [new_message]

    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        temperature=0.7,
        stream=False,
    )

    response = completion.choices[0].message.content
    user_history.append({"role": "user", "content": user_input})
    user_history.append({"role": "assistant", "content": response})

    log_interaction(user_input, response)

    return jsonify({"response": response, "history": user_history})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
