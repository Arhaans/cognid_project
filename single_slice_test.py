#!/usr/bin/env python3
import sys
import os

# Add LLaVA-Med to path
sys.path.append("/cs/home/psaas6/LLaVA-Med")

import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
from datetime import datetime

def load_llava_med():
    """Load LLaVA-Med model"""
    model_path = "microsoft/llava-med-v1.5-mistral-7b"
    
    print("Loading LLaVA-Med model...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        load_4bit=True,
        device_map="auto"
    )
    
    return tokenizer, model, image_processor, context_len

def generate_neuroradiology_report():
    """Generate formal neuroradiology report using LLaVA-Med"""
    
    # Load model
    tokenizer, model, image_processor, context_len = load_llava_med()
    
    # Path to your slice 75
    slice_path = "/cs/home/psaas6/Data/ExtractedSlices/CogNID010_1/axial/axial_slice_075.png"
    
    # Enhanced neuroradiologist prompt
    prompt = """You are a neuroradiologist with expertise in neurodegenerative disorders. Analyze this brain MRI slice for signs of neurodegeneration, atrophy, or abnormalities consistent with Alzheimer's disease or any neurodegenerative disease. 

Please provide a detailed radiological report that includes:
1. Description of visible anatomical structures
2. Assessment of ventricular size and morphology
3. Evaluation of cortical thickness and any atrophy
4. Identification of any abnormal signal intensities or lesions
5. Overall impression and differential diagnosis considerations
6. Recommendations for further evaluation if needed

Format your response as a formal radiological report."""
    
    # Load and process image
    image = Image.open(slice_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    
    # Prepare conversation
    conv = conv_templates["llava_med"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + prompt)
    conv.append_message(conv.roles[1], None)
    prompt_formatted = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(
        prompt_formatted, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    
    # Generate response
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.1,  # Lower temperature for more focused medical analysis
            max_new_tokens=1024,  # More tokens for detailed report
            use_cache=True
        )
    
    # Decode response
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    response = response.split("ASSISTANT:")[-1].strip()
    
    # Format and display results
    print("\n" + "="*80)
    print("🧠 NEURORADIOLOGY REPORT - LLaVA-Med Analysis")
    print("="*80)
    print(f"Patient ID: CogNID010_1")
    print(f"Image: axial_slice_075.png (256×256)")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"AI System: LLaVA-Med v1.5")
    print("-"*80)
    print("\nRADIOLOGICAL ANALYSIS:")
    print(response)
    print("\n" + "="*80)
    
    # Save report to file
    report_filename = f"CogNID010_1_neuroradiology_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w') as f:
        f.write(f"NEURORADIOLOGY REPORT - LLaVA-Med Analysis\n")
        f.write(f"Patient ID: CogNID010_1\n")
        f.write(f"Image: axial_slice_075.png (256×256)\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"AI System: LLaVA-Med v1.5\n")
        f.write("-"*80 + "\n\n")
        f.write("RADIOLOGICAL ANALYSIS:\n")
        f.write(response)
    
    print(f"\n📄 Report saved to: {report_filename}")
    
    return response

if __name__ == "__main__":
    generate_neuroradiology_report()
