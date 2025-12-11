import os
import traceback
from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Check if Key exists
HF_API_KEY = os.environ.get("HF_API_KEY")
if not HF_API_KEY:
    print("⚠️ WARNING: HF_API_KEY is missing from Environment Variables!")

# Initialize Client
client = InferenceClient(token=HF_API_KEY, timeout=120)


@app.route("/process-voice", methods=["POST"])
def process_voice():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid payload"}), 400

        voice_prompt = data.get("prompt", "")

        # Simple debugging print
        print(f"Processing: {voice_prompt}")

        prompt_template = (
            "Task: Translate natural language to SQL.\n\n"
            "Input: Show me users from London\n"
            "SQL: SELECT * FROM users WHERE city = 'London'\n\n"
            "Input: Count the number of products with price over 50\n"
            "SQL: SELECT COUNT(*) FROM products WHERE price > 50\n\n"
            f"Input: {voice_prompt}\n"
            "SQL: "
        )

        generated_sql = client.text_generation(
            prompt_template,
            model="google/flan-t5-large",
            max_new_tokens=150,
            temperature=0.1,
        )

        return jsonify({"status": "success", "message": f"SQL: {generated_sql}"}), 200

    except Exception as e:
        # Print full error to Render Logs
        traceback.print_exc()
        # Return the specific error type to the user (repr)
        return jsonify({"status": "error", "message": f"Server Error: {repr(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
