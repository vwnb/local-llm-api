import json
import traceback
import ollama
import subprocess
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import logging
import librosa
import numpy as np
import json

app = Flask(__name__)
CORS(app)

model_name = "gemma3"

logging.basicConfig(level=logging.INFO)

def extract_musical_features(path):

    y, sr = librosa.load(path, sr=None, mono=True)
    y = librosa.util.normalize(y)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    logging.info(f"Extracted tempo")

    # Timbre / spectral
    brightness = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    energy = float(np.mean(librosa.feature.rms(y=y)))
    logging.info(f"Extracted timbre / spectral")

    # Adjust hop length: faster tempo → finer resolution
    hop_length = int(np.clip(512 * (120.0 / max(tempo, 60.0)), 256, 1024))
    hop_length = int(round(hop_length / 64) * 64)  # Ensure multiple of 64 for CQT

    # Adjust k-number of sections: longer tracks or slower tempo → more structure
    duration = librosa.get_duration(y=y, sr=sr)
    num_sections_k = int(np.clip((duration / 60.0) * (tempo / 120.0) + 3, 3, 6))

    # Key estimation
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_mean = chroma.mean(axis=1)
    key_index = int(np.argmax(chroma_mean))
    key_confidence = float(np.max(chroma_mean) / np.sum(chroma_mean))
    key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key_index]
    logging.info(f"Extracted key")

    # MFCCs / spectral envelope
    n_mfcc = int(np.clip(13 + (brightness / 1000.0), 13, 30)) # more complex spectrum → more coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_means = mfcc.mean(axis=1).tolist()
    mfcc_stds = mfcc.std(axis=1).tolist()
    logging.info(f"Extracted MFCCs")

    # Structural segmentation
    R = librosa.segment.recurrence_matrix(mfcc, sym=True, mode='affinity')
    embedding = librosa.segment.recurrence_to_lag(R, axis=1)
    boundaries = librosa.segment.agglomerative(embedding, k=num_sections_k)
    times = librosa.frames_to_time(boundaries, sr=sr, hop_length=hop_length)
    section_durations = np.diff(times)
    section_summary = {
        "num_sections": int(len(times) - 1),
        "section_boundaries_sec": [float(t) for t in times],
        "section_durations_sec": [float(d) for d in section_durations]
    }
    logging.info(f"Extracted structure")

    # Harmonic / percussive energy
    D = librosa.stft(y, hop_length=hop_length)
    margin = np.clip(2.0 + (brightness / 5000.0), 1.0, 5.0) # brighter (more percussive) → higher separation margin
    harmonic, percussive = librosa.decompose.hpss(D, margin=margin)
    h_y = librosa.istft(harmonic)
    p_y = librosa.istft(percussive)
    hpss_features = {
        "harmonic_energy": float(np.mean(librosa.feature.rms(y=h_y))),
        "percussive_energy": float(np.mean(librosa.feature.rms(y=p_y)))
    }
    logging.info(f"Extracted harmonic / percussive energy")

    return json.dumps({
        "tempo_bpm": float(tempo),
        "key": key,
        "key_confidence": key_confidence,
        "brightness": brightness,
        "spectral_contrast": spectral_contrast,
        "energy": energy,
        "sections": section_summary,
        "mfcc_means": mfcc_means,
        "mfcc_stds": mfcc_stds,
        "hpss_features": hpss_features
    }, indent=2)

def generate_feedback(model_name, prompt):
    try:
        reply = ollama.chat(model=model_name, options={"num_predict": 200}, messages=[
            {
                "role": "system",
                "content": (
                    "You are 風鈴 AI, a music feedback chatbot. "
                    "Analyze this song’s mood, rhythm, and timbral qualities based on the feature data."
                    "Respond in 3–4 sentences max."
                    "Conclude with something qualitative - do you think the song is objectively interesting or pleasant?"
                )
            },
            {"role": "user", "content": prompt}
        ])
        message = reply.get("message", {})
        content = message.get("content", "")

        if not content:
            raise ValueError("No content returned from model.")

        response = make_response(content, 200)

    except Exception as e:
        print("Error in generate_feedback:", traceback.format_exc())
        response = make_response(
            jsonify({"error": str(e)}),
            500
        )

    return response

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
    
    filename = request.args.get('filename', '')
    if not filename:
        return make_response("Missing 'filename' parameter", 400)

    features_str = extract_musical_features(filename)

    logging.info(f"Question: {question}")
    logging.info(f"Filename: {filename}")
    logging.info(f"Musical features extracted: {features_str}")

    prompt = f"Context:\n{features_str}\n\nQuestion: {question}"

    response = generate_feedback(model_name, prompt)
    response.headers["Access-Control-Allow-Origin"] = "*"
    
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)