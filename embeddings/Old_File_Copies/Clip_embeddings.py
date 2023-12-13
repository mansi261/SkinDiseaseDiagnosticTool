import pandas as pd
import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# Function to load an image from a URL
def load_image_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=5)  # Increased timeout
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.Timeout:
        print(f"Timeout occurred for {url}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"URL not found (404): {url}")
        else:
            print(f"HTTP error occurred for {url}: {e}")
    except requests.RequestException as e:
        print(f"Error loading image from {url}: {e}")
    return None

# Function to process a batch of images
def process_images(batch):
    results = []
    for url in batch:
        image = load_image_from_url(url)
        if image is not None:
            inputs = processor(images=image, return_tensors="pt")
            outputs = model.get_image_features(**inputs)
            results.append(outputs.detach().cpu().numpy()[0])
    return results

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load the DataFrame containing the URLs
df = pd.read_csv('/Users/mansivyas/RA-MultiModel-ReducedSpecifciation/embeddings/skindisease_filtered_cleaned.csv')  # Replace with your dataset path

# Batching URLs for parallel processing
batch_size = 20  # Adjust based on your system's capability
url_batches = [df['url'][i:i+batch_size] for i in range(0, len(df), batch_size)]

embeddings = []

# Process the images in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers based on your system
    for batch_results in executor.map(process_images, url_batches):
        embeddings.extend(batch_results)

# Assuming 'label' column exists in your DataFrame
# Creating a DataFrame for embeddings and adding the 'label' column
embeddings_df = pd.DataFrame(embeddings)
embeddings_df['label'] = df['label']

# Save the embeddings to a CSV file
embeddings_df.to_csv('skin_disease_embeddings.csv', index=False)
