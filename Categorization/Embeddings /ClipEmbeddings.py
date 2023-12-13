import pandas as pd
import torch
import clip
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO

# Load the CLIP model and preprocessing function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Function to load and preprocess an image from a URL
def load_and_preprocess_image(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200 or 'image' not in response.headers.get('Content-Type', ''):
        raise ValueError(f"Invalid response for URL {url}. Status Code: {response.status_code}, Content-Type: {response.headers.get('Content-Type')}")

    try:
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(f"Cannot identify image file from URL: {url}")

    # Apply the preprocessing function from CLIP
    image = preprocess(image).unsqueeze(0).to(device)
    return image

# Load your dataset
df = pd.read_csv('/Users/mansivyas/RA-MultiModel-ReducedSpecifciation/Categorization/nine_categorization_40.csv')  # Update this path with your actual file path

# Extract embeddings and labels
embeddings = []
nine_partition_labels = []

for _, row in df.iterrows():
    try:
        # Load and preprocess the image
        image = load_and_preprocess_image(row['image_url'])  # Ensure your dataset has a column 'image_url'

        # Disable gradient calculation for efficiency and inference purpose
        with torch.no_grad():
            # Forward pass through the model
            embedding = model.encode_image(image)

        # Convert to numpy and store
        embeddings.append(embedding.cpu().numpy().flatten())
        nine_partition_labels.append(row['nine_partition_label'])  # Ensure your dataset has a column 'nine_partition_label'
    except Exception as e:
        print(f"Error processing image {row['image_url']}: {e}")

# Convert embeddings and labels to a DataFrame
embeddings_df = pd.DataFrame(embeddings)
embeddings_df['nine_partition_label'] = nine_partition_labels

# Save embeddings to a CSV file
embeddings_df.to_csv('clip_embeddings_nine_partition.csv', index=False)
