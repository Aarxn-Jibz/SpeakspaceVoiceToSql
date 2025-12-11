import os
import sys
import traceback
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_NAME = "cssupport/t5-small-awesome-text-to-sql"

print(f"Loading model: {MODEL_NAME}...")

# Load model immediately. If this fails, the app should crash so you know why.
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print("Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model. {e}")
    # Exit the program because it cannot function without the model
    sys.exit(1)


# --- API ENDPOINT ---
@app.route("/process-voice", methods=["POST"])
def process_voice():
    try:
        # 1. Parse Data
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid payload"}), 400

        voice_prompt = data.get("prompt", "")
        print(f"Received prompt: {voice_prompt}")

        # 2. Pre-process Input
        input_text = f"translate to SQL: {voice_prompt}"

        # 3. Run Inference
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True,
        )

        # 4. Decode Output
        generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated SQL: {generated_sql}")

        # 5. Return Response
        return jsonify({"status": "success", "message": f"SQL: {generated_sql}"}), 200

    except Exception as e:
        # Print the full error traceback to the terminal for debugging
        print("Error processing request:")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Server Error: {str(e)}"}), 500


# --- RUN SERVER ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
