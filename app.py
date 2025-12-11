import os
import traceback
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
API_URL = (
    "https://api-inference.huggingface.co/models/cssupport/t5-small-awesome-text-to-sql"
)

# Get token from environment or string (for testing)
HF_API_KEY = os.environ.get("HF_API_KEY")
headers = {"Authorization": f"Bearer {HF_API_KEY}"}


def query_huggingface(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


@app.route("/process-voice", methods=["POST"])
def process_voice():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid payload"}), 400

        voice_prompt = data.get("prompt", "")
        print(f"Received prompt: {voice_prompt}")

        # 1. Prepare Payload
        payload = {
            "inputs": f"translate to SQL: {voice_prompt}",
            "options": {"wait_for_model": True},
        }

        # 2. Call API
        output = query_huggingface(payload)

        # 3. Parse Response
        generated_sql = "Error generating SQL"

        # Hugging Face returns a list like [{'generated_text': 'SELECT...'}]
        if isinstance(output, list) and len(output) > 0:
            generated_sql = output[0].get("generated_text", generated_sql)

        # Handle errors (e.g. model loading)
        elif isinstance(output, dict) and "error" in output:
            print(f"API Error: {output}")
            return jsonify(
                {"status": "error", "message": "AI is warming up, try again in 10s"}
            ), 503

        print(f"Generated SQL: {generated_sql}")

        return jsonify({"status": "success", "message": f"SQL: {generated_sql}"}), 200

    except Exception as e:
        print("Error processing request:")
        traceback.print_exc()
        return jsonify({"status": "error", "message": "Internal Server Error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
