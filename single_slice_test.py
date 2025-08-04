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

def load_llava_med_no_quant():
    """Load LLaVA-Med with multiple fallback strategies and multi-GPU support"""
    model_path = "microsoft/llava-med-v1.5-mistral-7b"

    try:
        # First try: FP16 on GPU(s)
        print("Trying FP16 on GPU(s)...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device_map="auto"
        )

        # Enable multi-GPU if available
        if torch.cuda.device_count() > 1:
            print(f"✔️ Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)

        print("✔️ Model loaded on GPU(s) (FP16)")

    except Exception as e:
        # Fallback: CPU execution
        print(f"GPU failed ({e}), trying CPU...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),  
            device_map="cpu"
        )
        print("✔️ Model loaded on CPU (FP32)")

    return tokenizer, model, image_processor, context_len

def analyze_brain_slice():
    """Analyze CogNID010_1 brain slice with LLaVA-Med"""
    
    # Load model
    tokenizer, model, image_processor, context_len = load_llava_med_no_quant()
    
    # Path to slice
    slice_path = "/cs/home/psaas6/cognid_project/axial_slice_075.png"
    
    if not os.path.exists(slice_path):
        print(f"❌ File not found: {slice_path}")
        print("Available files in the project folder:")
        project_dir = os.path.dirname(slice_path)
        for f in sorted(os.listdir(project_dir)):
            print("  ", f)
        return

    # Neuroradiologist prompt
    prompt = """You are a neuroradiologist with expertise in neurodegenerative disorders. Analyze this brain MRI slice for signs of neurodegeneration, atrophy, or abnormalities consistent with Alzheimer's disease or any neurodegenerative disease. 

Please provide a detailed radiological report that includes:
1. Description of visible anatomical structures
2. Assessment of ventricular size and morphology
3. Evaluation of cortical thickness and any atrophy
4. Identification of any abnormal signal intensities or lesions
5. Overall impression and differential diagnosis considerations

Format your response as a formal radiological report."""
    
    print("Loading image...")
    image = Image.open(slice_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.module.config if isinstance(model, torch.nn.DataParallel) else model.config)
    image_tensor = [img.to(model.device if not isinstance(model, torch.nn.DataParallel) else model.module.device, dtype=torch.float16) for img in image_tensor]

    print("Preparing analysis...")
    print("Available conversation templates:", list(conv_templates.keys()))
    conv = conv_templates["llava_v1"].copy() 
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + prompt)
    conv.append_message(conv.roles[1], None)
    prompt_formatted = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_formatted, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0)

    device = model.device if not isinstance(model, torch.nn.DataParallel) else model.module.device
    input_ids = input_ids.to(device)

    print("Generating neuroradiology report...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True
        )

    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    response = response.split("ASSISTANT:")[-1].strip()

    print("\n" + "="*80)
    print("🧠 NEURORADIOLOGY REPORT - LLaVA-Med Analysis")
    print("="*80)
    print(f"Patient ID: CogNID010_1")
    print(f"Image: axial_slice_075.png (256×256)")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*80)
    print("\nRADIOLOGICAL ANALYSIS:")
    print(response)
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_brain_slice()
