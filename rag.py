import json
import chromadb
from chromadb.utils import embedding_functions
import ollama
import subprocess
from flask import Flask, request, make_response
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

model_name = "gemma3"

try:
    ollama.chat(model=model_name, messages=[{"role": "user", "content": "Hello"}])
except Exception as e:
    print(f"Model '{model_name}' not found. Pulling now...")
    subprocess.run(["ollama", "pull", model_name], check=True)

vectors_file = "vectors.json"

documents = []
ids = []

with open(vectors_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for i, row in enumerate(data):
    # Flatten dict to a single string, e.g. "column1: value1 column2: value2 ..."
    document = " ".join([f"{k}: {v}" for k, v in row.items()])
    documents.append(document)
    ids.append(str(i))

client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(name="jazz_vectors", embedding_function=embedding_fn)

if collection.count() == 0:
    collection.add(documents=documents, ids=ids)
else:
    print("Collection already contains documents.")

logging.basicConfig(level=logging.INFO)

@app.before_request
def log_request_info():
    logging.info(f"Request: {request.method} {request.url}")
    logging.info(f"Headers: {dict(request.headers)}")
    logging.info(f"Body: {request.get_data()}")

@app.route('/ask', methods=['GET', 'OPTIONS'])
def handle_ask():
    if request.method == 'OPTIONS':
        response = make_response('', 204)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'User-Agent, ngrok-skip-browser-warning, Content-Type'
        return response

    question = request.args.get('question', '')
    if not question:
        return make_response("Missing 'question' parameter", 400)

    results = collection.query(query_texts=[question], n_results=3)

    if not results or not results['documents'][0]:
        return make_response("No relevant documents found", 404)

    context = "\n".join(results['documents'][0])
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    reply = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are 風鈴 AI, a jazz chatbot."
                    "Answer based only on the context."
                )
            },
            {"role": "user", "content": prompt}
        ]
    )

    content = reply.get('message', {}).get('content', '')
    response = make_response(content, 200)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)