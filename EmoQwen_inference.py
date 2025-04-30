#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EmoLLMs Single File Inference Script - Using VLLM for accelerated inference
Usage:
    python single_file_inference.py
"""

import os
import random
import numpy as np
import torch
from typing import List, Dict, Any

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import VLLM related libraries
from vllm import LLM, SamplingParams
# Import transformers library to use tokenizer
from transformers import AutoTokenizer

# ================ Configuration Parameters ================
# Model configuration
MODEL_NAME_OR_PATH = "./model_Emo-Qwen"  # Local model path
USE_VLLM = True  # Whether to use vllm for inference

# Generation configuration
MAX_NEW_TOKENS = 256  # Maximum number of new tokens to generate
TEMPERATURE = 0.0  # Sampling temperature
TOP_P = 0.6  # Probability threshold for nucleus sampling
TOP_K = 30  # K value for top-k sampling
SEED = 42  # Random seed

# GPU configuration
GPU_MEMORY_UTILIZATION = 0.9  # GPU memory utilization
TENSOR_PARALLEL_SIZE = 1  # Tensor parallel size

# ================ Utility Functions ================
def set_seed(seed=42):
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

def run_vllm_inference(model, tokenizer, prompts: List[str], sampling_params):
    """Run inference using VLLM"""
    # Format prompts using tokenizer's apply_chat_template method
    formatted_prompts = []
    for prompt in prompts:
        # Build chat format
        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            chat, 
            add_generation_prompt=True,
            tokenize=False
        )
        formatted_prompts.append(formatted_prompt)
    
    # Print prompts
    for i, prompt in enumerate(formatted_prompts):
        print(f"\nPrompt #{i+1}: '{prompt[:50]}...'")
    
    # Generate using VLLM
    outputs = model.generate(formatted_prompts, sampling_params)
    
    # Process results
    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        print(f"\n--- Generation Result #{i+1} ---")
        print(generated_text)
        print("----------------\n")
        results.append(generated_text)
    
    return results

def run_test_prompts(model, tokenizer, sampling_params):
    """Run multiple test prompts"""
    test_prompts = [
        # Example 1: Emotion intensity
        "Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent the intensity of emotion E expressed in the text.\nText: @CScheiwiller can't stop smiling 😆😆😆\nEmotion: joy\nIntensity Score:",
        
        # Example 2: Sentiment strength
        "Task: Evaluate the valence intensity of the writer's mental state based on the text, assigning it a real-valued score from 0 (most negative) to 1 (most positive).\nText: Happy Birthday shorty. Stay fine stay breezy stay wavy @daviistuart 😘\nIntensity Score:",
        
        # Example 3: Sentiment classification
        "Task: Categorize the text into an ordinal class that best characterizes the writer's mental state, considering various degrees of positive and negative sentiment intensity. 3: very positive mental state can be inferred. 2: moderately positive mental state can be inferred. 1: slightly positive mental state can be inferred. 0: neutral or mixed mental state can be inferred. -1: slightly negative mental state can be inferred. -2: moderately negative mental state can be inferred. -3: very negative mental state can be inferred\nText: Beyoncé resentment gets me in my feelings every time. 😩\nIntensity Class:",
        
        # Example 4: Emotion classification
        "Task: Categorize the text's emotional tone as either 'neutral or no emotion' or identify the presence of one or more of the given emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust).\nText: Whatever you decide to do make sure it makes you #happy.\nThis text contains emotions:"
    ]
    
    # Run inference using VLLM
    results = run_vllm_inference(model, tokenizer, test_prompts, sampling_params)
    return results

def main():
    """Main function"""
    # Set random seed
    set_seed(SEED)
    
    # Initialize tokenizer
    tokenizer = setup_tokenizer()
    
    # Initialize VLLM model
    vllm_model = setup_vllm_model()
    
    # Set sampling parameters
    sampling_params = setup_sampling_params()
    
    # Run test prompts
    print("\n=== Running Test Prompts ===")
    test_results = run_test_prompts(vllm_model, tokenizer, sampling_params)
    
    print("\nAll test prompts have been processed")
    print(f"VLLM accelerated inference completed!")

if __name__ == "__main__":
    main()