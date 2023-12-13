import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO

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

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)

# Initialize a pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()


df = pd.read_csv('/Users/mansivyas/RA-MultiModel-ReducedSpecifciation/Categorization/nine_partition_embeddings_40.csv')

embeddings = []
nine_partition_labels = []

for _, row in df.iterrows():
    try:
        image = load_and_preprocess_image(row['url'])
        image = image.unsqueeze(0)

      
        with torch.no_grad():
            # Forward pass through the model
            embedding = model(image)

        # Convert to numpy and store
        embeddings.append(embedding.cpu().numpy().flatten())
        nine_partition_labels.append(row['nine_partition_label'])
    except Exception as e:
        print(f"Error processing image {row['image_url']}: {e}")

# Convert embeddings and labels to a DataFrame
embeddings_df = pd.DataFrame(embeddings)
embeddings_df['nine_partition_label'] = nine_partition_labels

# Save embeddings to a CSV file
embeddings_df.to_csv('embeddings_nine_partition.csv', index=False)
