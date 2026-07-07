"""
Project 4: Image Recognition (Basic)
DecodeLabs AI Training Kit
"""

from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

print("🚀 DecodeLabs Project 4: Image Recognition")
print("="*60)

# Load pre-trained image classification model
print("\nLoading pre-trained model... (first time may take a minute)")
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Option 1: Use a sample image from URL
def recognize_from_url(image_url):
    print(f"\n📸 Analyzing image from: {image_url}")
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
    results = classifier(image)
    
    print("\n🔍 Recognition Results:")
    print("-" * 40)
    for result in results[:5]:  # Top 5 predictions
        print(f"{result['label']:30} : {result['score']:.4f} ({result['score']*100:.2f}%)")

# Option 2: Use local image (recommended for submission)
def recognize_from_local(image_path):
    print(f"\n📸 Analyzing local image: {image_path}")
    image = Image.open(image_path)
    
    results = classifier(image)
    
    print("\n🔍 Recognition Results:")
    print("-" * 40)
    for result in results[:5]:
        print(f"{result['label']:30} : {result['score']:.4f} ({result['score']*100:.2f}%)")

# ====================== RUN EXAMPLES ======================
if __name__ == "__main__":
    # Example 1: From URL
    sample_url = "https://upload.wikimedia.org/wikipedia/commons/3/3f/Walking_tiger_female.jpg"
    recognize_from_url(sample_url)
    
    print("\n" + "="*60)
    print("✅ Project 4 Completed!")
    print("   Used: Hugging Face ViT model for Image Classification")
