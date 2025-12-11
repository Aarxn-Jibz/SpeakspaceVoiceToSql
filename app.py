import os
import traceback
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
# Let's go back to the standard URL structure, but keep the router domain
API_URL = "https://router.huggingface.co/models/google/flan-t5-large"

HF_API_KEY = os.environ.get("HF_API_KEY")
headers = {"Authorization": f"Bearer {HF_API_KEY}"}


def query_huggingface(payload):
    print(f"⚡ Sending request to: {API_URL}")
    response = requests.post(API_URL, headers=headers, json=payload)

    # DEBUGGING: Print exactly what HF sent back
    print(f"📥 HF Status Code: {response.status_code}")
    print(f"📥 HF Raw Response: {response.text}")

    # Only try to parse JSON if the status is good
    if response.status_code != 200:
        return {"error": f"HF Error {response.status_code}", "raw": response.text}

    return response.json()


@app.route("/process-voice", methods=["POST"])
def process_voice():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid payload"}), 400

        voice_prompt = data.get("prompt", "")
        print(f"🎤 Received prompt: {voice_prompt}")

        prompt_template = (
            "Task: Translate natural language to SQL.\n\n"
            "Input: Show me users from London\n"
            "SQL: SELECT * FROM users WHERE city = 'London'\n\n"
            "Input: Count the number of products with price over 50\n"
            "SQL: SELECT COUNT(*) FROM products WHERE price > 50\n\n"
            f"Input: {voice_prompt}\n"
            "SQL: "
        )

        payload = {
            "inputs": prompt_template,
            "options": {"wait_for_model": True},
        }

        output = query_huggingface(payload)

        # Logic to extract text safely
        generated_sql = "Error generating SQL"

        # Check if output is the list we expect
        if isinstance(output, list) and len(output) > 0:
            generated_sql = output[0].get("generated_text", generated_sql)
        # Check if output is an error dictionary
        elif isinstance(output, dict) and ("error" in output or "raw" in output):
            print(f"⚠️ API Issue: {output}")
            return jsonify({"status": "error", "message": f"AI Error: {output}"}), 500

        print(f"🤖 Generated SQL: {generated_sql}")

        return jsonify({"status": "success", "message": f"SQL: {generated_sql}"}), 200

    except Exception as e:
        print("🔥 Critical Exception:")
        traceback.print_exc()
        return jsonify({"status": "error", "message": "Internal Server Error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
