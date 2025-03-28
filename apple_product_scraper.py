import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
from typing import List, Dict
import uuid
from pathlib import Path

# Configuration
OUTPUT_DIR = "apple_products"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
PRODUCTS_FILE = os.path.join(OUTPUT_DIR, "products.json")
BASE_URL = "https://www.apple.com"
PRODUCT_CATEGORIES = [
    "/iphone/",
    "/ipad/",
    "/mac/",
    "/watch/",
    "/airpods/",
    "/homepod/",
    "/accessories/"
]

def setup_directories():
    """Create necessary directories for storing data"""
    os.makedirs(IMAGE_DIR, exist_ok=True)

def download_image(url: str, product_id: str) -> str:
    """Download product image and return local path"""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            ext = url.split('.')[-1].split('?')[0]
            image_path = os.path.join(IMAGE_DIR, f"{product_id}.{ext}")
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return image_path
    except Exception as e:
        print(f"Failed to download image: {e}")
    return ""

def generate_mock_apple_products() -> List[Dict]:
    """Generate mock Apple products for demonstration"""
    products = []
    categories = ["iPhone", "iPad", "Mac", "Watch", "AirPods"]
    
    for i in range(1, 101):
        category = categories[i % len(categories)]
        product_id = str(uuid.uuid4())
        
        products.append({
            "id": product_id,
            "title": f"Apple {category} {i}",
            "description": f"Latest generation {category} with advanced features",
            "price": f"${799 + (i * 10)}.00",
            "image_url": f"https://example.com/images/{category.lower()}_{i}.jpg",
            "image_path": "",
            "category": category,
            "scraped_at": datetime.now().isoformat()
        })
    
    return products

def scrape_apple_products() -> List[Dict]:
    """Get Apple products (using mock data for this demo)"""
    return generate_mock_apple_products()

def save_products(products: List[Dict]):
    """Save scraped products to JSON file"""
    with open(PRODUCTS_FILE, 'w') as f:
        json.dump(products, f, indent=2)

def insert_to_vector_db(products: List[Dict]):
    """Insert products into vector database using the FastAPI endpoint"""
    for product in products:
        try:
            document = {
                "text": f"{product['title']}. {product['description']}",
                "metadata": {
                    "id": product["id"],
                    "title": product["title"],
                    "price": product["price"],
                    "category": product["category"],
                    "image_url": product["image_url"],
                    "image_path": product["image_path"],
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
    setup_directories()
    print("Scraping Apple products...")
    products = scrape_apple_products()
    print(f"Scraped {len(products)} products")
    save_products(products)
    print("Inserting products to vector database...")
    insert_to_vector_db(products)
    print("Done!")