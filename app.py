import os
import sys
import traceback
import gc
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

MODEL_NAME = "cssupport/t5-small-awesome-text-to-sql"

print(f"Loading model: {MODEL_NAME}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load heavy model first
    _model_heavy = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Quantize to 8-bit (shrinks memory usage by ~50%)
    model = torch.quantization.quantize_dynamic(
        _model_heavy, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Cleanup heavy model to free RAM
    del _model_heavy
    gc.collect()

    print("Model loaded and quantized successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model. {e}")
    sys.exit(1)


@app.route("/process-voice", methods=["POST"])
def process_voice():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid payload"}), 400

        voice_prompt = data.get("prompt", "")
        print(f"Received prompt: {voice_prompt}")

        input_text = f"translate to SQL: {voice_prompt}"

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

        generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated SQL: {generated_sql}")

        return jsonify({"status": "success", "message": f"SQL: {generated_sql}"}), 200

    except Exception as e:
        print("Error processing request:")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Server Error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
