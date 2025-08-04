#!/usr/bin/env python3
"""
Neurodegeneration analysis using LLaVA-Med for brain MRI slice
"""
import torch
import sys
import os
from PIL import Image
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

def analyze_brain_slice_neurodegeneration(image_path, model_path):
    """
    Detailed neurodegeneration analysis of brain MRI slice using LLaVA-Med
    """
    print(f"Loading LLaVA-Med model from: {model_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path, 
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    
    # Load and process image
    print(f"Loading brain MRI image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # Expert prompt for neurodegeneration analysis
    neurodegeneration_prompt = (
        "You are a neuroradiologist with expertise in neurodegenerative disorders. "
        "Analyze this brain MRI slice for signs of neurodegeneration, atrophy, or abnormalities "
        "consistent with Alzheimer's disease or any neurodegenerative disease.\n\n"
        "Please provide a detailed radiological report that includes:\n"
        "1. Description of visible anatomical structures\n"
        "2. Assessment of ventricular size and morphology\n"
        "3. Evaluation of cortical thickness and any atrophy\n"
        "4. Identification of any abnormal signal intensities or lesions\n"
        "5. Overall impression and differential diagnosis considerations\n\n"
        "Format your response as a formal radiological report."
    )
    
    # Prepare conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": neurodegeneration_prompt}
            ]
        }
    ]
    
    # Process inputs
    inputs = processor.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    
    # Generate response
    print("Generating neurodegeneration analysis...")
    print("=" * 80)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            max_new_tokens=1024,
            use_cache=True
        )
    
    # Decode response
    response = processor.decode(output_ids[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    image_path = "./axial_slice_075.png"
    model_path = "/cs/home/psaas6/models/llava-med-v1.5/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/f2f72301dc934e74948b5802c87dbc83d100e6bd"

    print("🧠 LLaVA-Med Neurodegeneration Analysis")
    print("=" * 80)
    print(f"Image: {os.path.abspath(image_path)}")
    print(f"Model: {model_path}")
    print(f"Analysis: Neurodegeneration & Alzheimer's Disease Assessment")
    print("=" * 80)

    if not os.path.exists(image_path):
        print(f"❌ Error: Brain MRI image not found at {image_path}")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"❌ Error: LLaVA-Med model not found at {model_path}")
        sys.exit(1)

    try:
        result = analyze_brain_slice_neurodegeneration(image_path, model_path)

        print("\n🏥 NEURORADIOLOGY REPORT")
        print("=" * 80)
        print(result)
        print("=" * 80)

        with open("neurodegeneration_report.txt", "w") as f:
            f.write("NEURORADIOLOGY REPORT - NEURODEGENERATION ANALYSIS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Image: {os.path.abspath(image_path)}\n")
            f.write(f"Model: LLaVA-Med v1.5\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(result)

        print(f"\n📄 Report saved to: neurodegeneration_report.txt")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)
