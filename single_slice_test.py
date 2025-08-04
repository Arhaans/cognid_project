#!/usr/bin/env python3
import sys
import os
import torch
from PIL import Image
from datetime import datetime

# Add LLaVA-Med to Python path
sys.path.append("/cs/home/psaas6/LLaVA-Med")

# LLaVA-Med imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

def load_llava_med_model():
    """Load LLaVA-Med with automatic device placement (multi-GPU supported via Transformers)"""
    model_path = "microsoft/llava-med-v1.5-mistral-7b"
    print("🔄 Loading model (device_map='auto')...")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device_map="auto"
    )

    print("✅ Model loaded")
    return tokenizer, model, image_processor, context_len

def analyze_brain_slice():
    """Run image-based prompt through LLaVA-Med and print formatted report"""

    tokenizer, model, image_processor, context_len = load_llava_med_model()
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load brain slice
    slice_path = "/cs/home/psaas6/cognid_project/axial_slice_075.png"
    if not os.path.exists(slice_path):
        print(f"❌ Image file not found: {slice_path}")
        return

    image = Image.open(slice_path).convert('RGB')
    model_config = model.config
    image_tensor = process_images([image], image_processor, model_config)
    image_tensor = image_tensor[0].to(model_device, dtype=torch.float32)  # ✅ Ensure single Tensor

    print("📷 Image tensor shape:", image_tensor.shape)

    # Construct prompt
    prompt = """You are a neuroradiologist with expertise in neurodegenerative disorders. Analyze this brain MRI slice for signs of neurodegeneration, atrophy, or abnormalities consistent with Alzheimer's disease or any neurodegenerative disease. 

Please provide a detailed radiological report that includes:
1. Description of visible anatomical structures
2. Assessment of ventricular size and morphology
3. Evaluation of cortical thickness and any atrophy
4. Identification of any abnormal signal intensities or lesions
5. Overall impression and differential diagnosis considerations

Format your response as a formal radiological report."""

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + prompt)
    conv.append_message(conv.roles[1], None)
    prompt_formatted = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_formatted, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model_device)

    # Generate report
    print("🧠 Generating neuroradiology report...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,  # ✅ Correct shape
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True
        )

    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    response = response.split("ASSISTANT:")[-1].strip()

    # Display formatted output
    print("\n" + "=" * 80)
    print("🧠 NEURORADIOLOGY REPORT - LLaVA-Med Analysis")
    print("=" * 80)
    print(f"Patient ID: CogNID010_1")
    print(f"Image: axial_slice_075.png (256×256)")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print("\nRADIOLOGICAL ANALYSIS:\n")
    print(response)
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_brain_slice()
