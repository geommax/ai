from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
USE_OLLAMA = os.getenv("USE_OLLAMA", "False").lower() == "true"
USE_TRANSFORMER = os.getenv("USE_TRANSFORMER", "False").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TRANSFORMER_MODEL = os.getenv("TRANSFORMER_MODEL", "llama3.2:latest")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for transformer model
tokenizer = None
model = None

def load_transformer_model():
    """Load transformer model and tokenizer"""
    global tokenizer, model
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        logger.info(f"Loading transformer model: {TRANSFORMER_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(TRANSFORMER_MODEL, trust_remote_code=True)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("Transformer model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load transformer model: {e}")
        return False

def generate_with_transformer(prompt, max_tokens=100):
    """Generate text using transformer model"""
    try:
        import torch
        
        if tokenizer is None or model is None:
            return "Error: Transformer model not loaded"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from the response
        response = response[len(prompt):].strip()
        
        return response if response else "I got it"
        
    except Exception as e:
        logger.error(f"Error generating with transformer: {e}")
        return f"Error: {str(e)}"

def generate_with_ollama(prompt, max_tokens=100):
    """Generate text using Ollama API"""
    try:
        import requests
        
        # Ollama API endpoint
        url = f"{OLLAMA_HOST}/api/generate"
        
        # Request payload
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        # Make request to Ollama
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        return result.get("response", "No response generated")
        
    except ImportError:
        logger.error("Requests library not available for Ollama integration")
        return "Error: Requests library not available"
    except Exception as e:
        logger.error(f"Error generating with Ollama: {e}")
        return f"Error: {str(e)}"

@app.route('/')
def index():
    return jsonify({
        "message": "LLM Backend Server Running",
        "use_ollama": USE_OLLAMA,
        "use_transformer": USE_TRANSFORMER,
        "ollama_model": OLLAMA_MODEL if USE_OLLAMA else None,
        "transformer_model": TRANSFORMER_MODEL if USE_TRANSFORMER else None
    })

@app.route('/generate', methods=['POST'])
def generate_response():
    start_time = time.time()
    
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 100)
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Generate response using transformer, Ollama, or simulation
        if USE_TRANSFORMER and tokenizer is not None and model is not None:
            response_text = generate_with_transformer(prompt, max_tokens)
        elif USE_OLLAMA:
            response_text = generate_with_ollama(prompt, max_tokens)
        else:
            # Simulate more realistic LLM processing time based on prompt length
            # This simulates that longer prompts take more time to process
            base_delay = 0.3  # Base 300ms delay
            prompt_length_factor = min(len(prompt) * 0.005, 0.7)  # Up to 700ms extra for very long prompts
            processing_delay = base_delay + prompt_length_factor
            
            # Add some random variation to simulate real-world variance
            import random
            variation = random.uniform(-0.1, 0.1)  # +/- 100ms variation
            processing_delay = max(0.1, processing_delay + variation)  # Minimum 100ms
            
            time.sleep(processing_delay)
            
            # Simple response for testing
            response_text = "I got it"
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return jsonify({
            "response": response_text,
            "latency": {
                "processing": round(processing_time * 1000)  # Convert to milliseconds
            }
        })
    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    if USE_TRANSFORMER:
        model_status = "available" if tokenizer is not None and model is not None else "unavailable"
        return jsonify({
            "status": "healthy",
            "use_transformer": USE_TRANSFORMER,
            "transformer_status": model_status,
            "transformer_model": TRANSFORMER_MODEL
        })
    elif USE_OLLAMA:
        try:
            import requests
            # Check if Ollama is running
            response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            ollama_status = "available" if response.status_code == 200 else "unavailable"
        except:
            ollama_status = "unavailable"
        return jsonify({
            "status": "healthy",
            "use_ollama": USE_OLLAMA,
            "ollama_status": ollama_status
        })
    else:
        return jsonify({
            "status": "healthy",
            "mode": "simulation"
        })

if __name__ == '__main__':
    logger.info("Starting LLM Backend Server")
    
    # Load transformer model if enabled
    if USE_TRANSFORMER:
        logger.info(f"Loading transformer model: {TRANSFORMER_MODEL}")
        if load_transformer_model():
            logger.info("Transformer model loaded successfully")
        else:
            logger.error("Failed to load transformer model")
            USE_TRANSFORMER = False
    
    if USE_OLLAMA:
        logger.info(f"Using Ollama with model: {OLLAMA_MODEL}")
        logger.info(f"Ollama host: {OLLAMA_HOST}")
    elif USE_TRANSFORMER:
        logger.info(f"Using transformer model: {TRANSFORMER_MODEL}")
    else:
        logger.info("Running in simulation mode")
    
    app.run(host='0.0.0.0', port=5002, debug=True)