import chromadb
from chromadb.utils import embedding_functions
import ollama
import subprocess
from flask import Flask, request, make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_name = "gemma3"

try:
    ollama.chat(model=model_name, messages=[{"role": "user", "content": "Hello"}])
except ollama.ResponseError:
    print("Model not found. Pulling "+model_name+" now...")
    subprocess.run(["ollama", "pull", model_name], check=True)

with open("vectors.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]

documents = lines
ids = [str(i) for i in range(len(documents))]

client = chromadb.Client()
collection = client.create_collection(
    "docs",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
)

collection.add(documents=documents, ids=ids)

@app.route('/ask', methods=['GET'])
def handle_get():
    question = request.args.get('question')

    query = "Ask a question: " + question

    results = collection.query(query_texts=[query])
    context = "\n".join(results['documents'][0])

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer based only on the context."

    reply = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    response = make_response(reply['message']['content'], 200)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['ngrok-skip-browser-warning'] = 'bojojoing'

    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

