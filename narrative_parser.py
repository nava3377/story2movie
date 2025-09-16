# File: narrative_parser.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def decompose_story_into_scenes(story_text, model, tokenizer):
    """Uses an LLM to break down a story into a structured list of scenes."""
    print("🧠 Decomposing story into scenes...")
    
    prompt = f"""
    You are an expert screenwriter. Your task is to read the following story and decompose it into a sequence of distinct visual scenes.
    For each scene, create a concise visual description suitable for a text-to-image AI, and identify the corresponding narration text.

    The story is:
    "{story_text}"

    Please format your response ONLY as a valid JSON array of objects with "scene_description" and "narration_text" keys.
    JSON Response:
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        json_part = response_text.split("JSON Response:")[1].strip()
        return json.loads(json_part)
    except (IndexError, json.JSONDecodeError) as e:
        print(f"❌ Error parsing the LLM's response. Raw response was:\n{response_text}")
        return None