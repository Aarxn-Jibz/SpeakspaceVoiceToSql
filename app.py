import os
import requests
import traceback
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
# We use Google's model because it is UNGATED (No license acceptance needed)
API_URL = "https://router.huggingface.co/models/google/flan-t5-large"

HF_API_KEY = os.environ.get("HF_API_KEY")
headers = {"Authorization": f"Bearer {HF_API_KEY}"}


def query_huggingface(payload):
    print(f"⚡ Sending to: {API_URL}")
    try:
        response = requests.post(API_URL, headers=headers, json=payload)

        # DEBUG: Print status to logs
        print(f"📥 Status: {response.status_code}")

        # 1. Handle "Loading" State
        if response.status_code == 503:
            return {"error": "warming_up", "raw": response.text}

        # 2. Handle Success
        if response.status_code == 200:
            return response.json()

        # 3. Handle Errors (404, 401, 500)
        # We return the RAW TEXT so you can see exactly what HF is saying
        return {"error": f"HF Error {response.status_code}", "raw": response.text}

    except Exception as e:
        return {"error": f"Network Error: {str(e)}"}


@app.route("/process-voice", methods=["POST"])
def process_voice():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid payload"}), 400

        voice_prompt = data.get("prompt", "")
        print(f"🎤 Prompt: {voice_prompt}")

        # Flan-T5 needs examples to know it should write SQL
        prompt_template = (
            "Task: Translate natural language to SQL.\n\n"
            "Input: Show me users from London\n"
            "SQL: SELECT * FROM users WHERE city = 'London'\n\n"
            "Input: Count the number of products with price over 50\n"
            "SQL: SELECT COUNT(*) FROM products WHERE price > 50\n\n"
            f"Input: {voice_prompt}\n"
            "SQL: "
        )

        payload = {"inputs": prompt_template, "options": {"wait_for_model": True}}

        output = query_huggingface(payload)

        # Logic to extract text safely
        generated_sql = "Error"

        # Success Case
        if isinstance(output, list) and len(output) > 0:
            generated_sql = output[0].get("generated_text", "Error").strip()

        # Error Case
        elif isinstance(output, dict) and "error" in output:
            error_msg = output.get("error")
            raw_text = output.get("raw", "")

            print(f"⚠️ API Issue: {error_msg}")

            if "warming_up" in str(error_msg):
                return jsonify(
                    {
                        "status": "error",
                        "message": "AI is warming up... Try again in 20s",
                    }
                ), 503

            # This will show you the exact error from HF if it fails
            return jsonify(
                {
                    "status": "error",
                    "message": f"AI Error: {error_msg}. Raw: {raw_text[:50]}...",
                }
            ), 500

        print(f"🤖 SQL: {generated_sql}")
        return jsonify({"status": "success", "message": f"SQL: {generated_sql}"}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Server Error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
