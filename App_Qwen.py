#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EmoQwen Flask Web Application - Using VLLM for accelerated inference
"""

from flask import Flask, render_template, request, jsonify
import os
import random
import numpy as np
import torch
import time
from typing import List, Dict, Any

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import VLLM related libraries
from vllm import LLM, SamplingParams
# Import transformers library to use tokenizer
from transformers import AutoTokenizer

# Initialize Flask application
app = Flask(__name__)

# ================ Configuration Parameters ================
# Model configuration
MODEL_NAME_OR_PATH = "./model_Emo-Qwen"  # Local model path
USE_VLLM = True  # Whether to use vllm for inference

# Generation configuration
MAX_NEW_TOKENS = 256  # Maximum number of new tokens to generate
TEMPERATURE = 0.0  # Sampling temperature
TOP_P = 0.9  # Probability threshold for nucleus sampling
TOP_K = 30  # K value for top-k sampling
SEED = 42  # Random seed

# GPU configuration
GPU_MEMORY_UTILIZATION = 0.9  # GPU memory utilization
TENSOR_PARALLEL_SIZE = 1  # Tensor parallel size

# Task descriptions
TASK_DESCRIPTIONS = {
    'emotion_intensity': "Assign a numerical value between 0 (least {emotion}) and 1 (most {emotion}) to represent the intensity of emotion {emotion} expressed in the text.",
    'sentiment_strength': "Evaluate the valence intensity of the writer's mental state based on the text, assigning it a real-valued score from 0 (most negative) to 1 (most positive).",
    'sentiment_classification': "Categorize the text into an ordinal class that best characterizes the writer's mental state, considering various degrees of positive and negative sentiment intensity. 3: very positive mental state can be inferred. 2: moderately positive mental state can be inferred. 1: slightly positive mental state can be inferred. 0: neutral or mixed mental state can be inferred. -1: slightly negative mental state can be inferred. -2: moderately negative mental state can be inferred. -3: very negative mental state can be inferred",
    'emotion_classification': "Categorize the text's emotional tone as either 'neutral or no emotion' or identify the presence of one or more of the given emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust)."
}

# ================ Utility Functions ================
def set_seed(seed=SEED):
    """Set random seed to ensure reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")

def setup_tokenizer():
    """Initialize tokenizer"""
    # Get the absolute path of the model
    abs_model_path = os.path.abspath(MODEL_NAME_OR_PATH)
    print(f"Loading tokenizer: {abs_model_path}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        abs_model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    
    print("Tokenizer loaded successfully!")
    return tokenizer

def setup_vllm_model():
    """Initialize VLLM model"""
    # Get the absolute path of the model
    abs_model_path = os.path.abspath(MODEL_NAME_OR_PATH)
    print(f"Loading model with VLLM: {abs_model_path}")
    
    # Initialize VLLM model
    vllm_model = LLM(
        model=abs_model_path,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=True,
        dtype="float16" if torch.cuda.is_available() else "float32",
    )
    
    print("VLLM model loaded successfully!")
    return vllm_model

def setup_sampling_params():
    """Set sampling parameters"""
    # Set VLLM sampling parameters
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_NEW_TOKENS,
        stop=None,  # Can set stop words
    )
    return sampling_params

def format_prompt(task_type, text, emotion=""):
    """Format prompt based on task type"""
    task = TASK_DESCRIPTIONS.get(task_type, '')
    
    # Replace {emotion} placeholders with the specified emotion
    if emotion:
        task = task.replace("{emotion}", emotion)
    
    # Build prompt based on task type
    if task_type == 'emotion_intensity':
        prompt = f"Task: {task}\nText: {text}\nEmotion: {emotion}\nIntensity Score:"
    elif task_type == 'sentiment_strength':
        prompt = f"Task: {task}\nText: {text}\nIntensity Score:"
    elif task_type == 'sentiment_classification':
        prompt = f"Task: {task}\nText: {text}\nIntensity Class:"
    elif task_type == 'emotion_classification':
        prompt = f"Task: {task}\nText: {text}\nThis text contains emotions:"
    
    return prompt

def run_inference(model, tokenizer, prompt, sampling_params):
    """Run inference using VLLM with proper chat template formatting"""
    # Build chat format
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat, 
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Generate using VLLM
    outputs = model.generate(formatted_prompt, sampling_params)
    
    # Extract generated text
    generated_text = outputs[0].outputs[0].text.strip()
    
    return generated_text

# ================ Initialize application ================
print("Initializing application...")
# Set random seed
set_seed()

# Initialize tokenizer and model
tokenizer = setup_tokenizer()
model = setup_vllm_model()

# Setup sampling parameters
sampling_params = setup_sampling_params()
print("Application initialized successfully!")

# ================ Flask routes ================
@app.route('/')
def index():
    """Render the index.html page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Process text analysis requests"""
    start_time = time.time()

    # Get request data
    data = request.json
    task_type = data.get('task_type', '')
    text = data.get('text', '')
    emotion = data.get('emotion', '')

    # Format prompt
    prompt = format_prompt(task_type, text, emotion)

    # Run inference with proper chat template formatting
    result = run_inference(model, tokenizer, prompt, sampling_params)

    # Calculate processing time
    processing_time = time.time() - start_time

    return jsonify({
        'result': result,
        'processing_time': f"{processing_time:.2f} seconds"
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)