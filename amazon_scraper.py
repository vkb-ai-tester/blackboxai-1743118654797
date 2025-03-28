import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import uuid
import time
from vector_db_qdrant import QdrantVectorDB
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict

# Configuration
OUTPUT_DIR = "amazon_products"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PRODUCTS_FILE = os.path.join(OUTPUT_DIR, "products.json")

# Initialize CLIP model for image embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_url: str) -> List[float]:
    """Generate embedding for product image using CLIP with dimension validation"""
    if not image_url:
        return [0.0] * 512
        
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        inputs = processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        embedding = image_features[0].tolist()
        if len(embedding) != 512:
            raise ValueError(f"Invalid embedding dimension: expected 512, got {len(embedding)}")
        return embedding
    except Exception as e:
        print(f"Error generating image embedding: {e}")
        return [0.0] * 512  # Return proper zero vector

def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text to max_length words to avoid sequence length issues"""
    words = text.split()
    if len(words) > max_length:
        return ' '.join(words[:max_length]) + '...'
    return text

def scrape_amazon_products(search_term: str = "laptop", max_pages: int = 3) -> List[Dict]:
    """Scrape real Amazon products for given search term"""
    products = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    for page in range(1, max_pages + 1):
        try:
            url = f"https://www.amazon.com/s?k={search_term}&page={page}"
            print(f"Scraping page {page}: {url}")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all product containers - updated selectors
            product_containers = soup.find_all('div', {'data-component-type': 's-search-result'})
            print(f"Found {len(product_containers)} product containers on page {page}")
            
            for item in product_containers:
                try:
                    # Extract product details with more robust checks
                    title_elem = item.find('h2')
                    if not title_elem:
                        print("Skipping item - no title found")
                        continue
                        
                    title = title_elem.text.strip()
                    product_url = f"https://www.amazon.com{title_elem.find('a')['href']}" if title_elem.find('a') else ""
                    
                    price_whole = item.find('span', {'class': 'a-price-whole'})
                    price_fraction = item.find('span', {'class': 'a-price-fraction'})
                    price = f"${price_whole.text}{price_fraction.text}" if price_whole and price_fraction else "Price not available"
                    
                    image_elem = item.find('img', {'class': 's-image'})
                    image_url = image_elem['src'] if image_elem else ""
                    
                    product_id = str(uuid.uuid4())
                    
                    products.append({
                        "id": product_id,
                        "title": title,
                        "description": f"Amazon product: {title}",
                        "price": price,
                        "image_url": image_url,
                        "product_url": product_url,
                        "scraped_at": datetime.now().isoformat()
                    })
                    
                    print(f"Added product: {title}")
                    
                except Exception as e:
                    print(f"Error parsing product: {str(e)}")
                    continue
                    
            # Respectful delay between pages
            time.sleep(5)  # Increased delay to avoid rate limiting
            
        except Exception as e:
            print(f"Error scraping page {page}: {str(e)}")
            continue
    
    print(f"Successfully scraped {len(products)} products")
    return products

def save_products(products: List[Dict]):
    """Save scraped products to JSON file"""
    with open(PRODUCTS_FILE, 'w') as f:
        json.dump(products, f, indent=2)

def insert_to_vector_db(products: List[Dict]):
    """Insert products with text and image embeddings into vector DB"""
    for product in products:
        try:
            # Get text and image embeddings with dimension validation
            truncated_text = truncate_text(f"{product['title']}. {product['description']}")
            # Get text embedding and verify dimension
            text_features = model.get_text_features(
                **processor(text=truncated_text, return_tensors="pt", padding=True, truncation=True)
            )
            text_embedding = text_features[0].tolist()
            actual_dim = len(text_embedding)
            print(f"Text embedding dimension: {actual_dim}")
            if actual_dim != 512:
                print(f"Warning: Expected 512 dimensions but got {actual_dim}, adjusting collection expectations")
            
            image_embedding = get_image_embedding(product['image_url'])
            
            document = {
                "text": f"{product['title']}. {product['description']}",
                "text_embedding": text_embedding,
                "image_embedding": image_embedding,
                "metadata": {
                    "id": product["id"],
                    "title": product["title"],
                    "price": product["price"],
                    "image_url": product["image_url"],
                    "scraped_at": product["scraped_at"]
                }
            }
            
            response = requests.post(
                "http://localhost:8000/documents",
                json=document
            )
            
            if response.status_code != 200:
                print(f"Failed to insert product {product['id']}: {response.text}")
                
        except Exception as e:
            print(f"Error inserting product {product['id']}: {e}")

if __name__ == "__main__":
    print("Scraping Amazon products...")
    products = scrape_amazon_products()
    print(f"Scraped {len(products)} products")
    save_products(products)
    print("Inserting products to vector database...")
    insert_to_vector_db(products)
    print("Done!")