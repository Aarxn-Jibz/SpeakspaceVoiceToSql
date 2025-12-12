import os
import requests
import traceback
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
# We use the OpenAI-Compatible Router
API_URL = "https://router.huggingface.co/v1/chat/completions"

# New Model: Qwen 2.5 Coder (State of the art for SQL)
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

HF_API_KEY = os.environ.get("HF_API_KEY")
headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}


def query_huggingface(payload):
    print(f"⚡ Sending to: {MODEL_ID}")
    try:
        response = requests.post(API_URL, headers=headers, json=payload)

        # DEBUG STATUS
        print(f"📥 Status: {response.status_code}")

        # 1. Handle "Loading"
        if response.status_code == 503:
            return {"error": "warming_up"}

        # 2. Handle Success
        if response.status_code == 200:
            return response.json()

        # 3. Handle Errors
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

        # Qwen works best with standard Chat format
        payload = {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a SQL expert. Output ONLY the SQL query. No explanation. No markdown.",
                },
                {"role": "user", "content": f"Convert to SQL: {voice_prompt}"},
            ],
            "max_tokens": 150,
            "temperature": 0.1,
            "stream": False,
        }

        output = query_huggingface(payload)

        generated_sql = "Error"

        if "choices" in output and len(output["choices"]) > 0:
            generated_sql = output["choices"][0]["message"]["content"].strip()
            # Clean up markdown if Qwen adds it
            generated_sql = (
                generated_sql.replace("```sql", "").replace("```", "").strip()
            )

        elif "error" in output:
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

            return jsonify(
                {
                    "status": "error",
                    "message": f"AI Error: {error_msg} Raw: {raw_text[:50]}...",
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
