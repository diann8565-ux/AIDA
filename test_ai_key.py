import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AI_API_KEY")
API_URL = os.getenv("AI_API_URL", "https://one.apprentice.cyou/api/v1/chat/completions")
MODEL = os.getenv("AI_MODEL", "gemini-2.5-flash")

print(f"--- AI Connection Test ---")
print(f"URL: {API_URL}")
print(f"Model: {MODEL}")
print(f"Key Loaded: {'Yes' if API_KEY and API_KEY != 'YOUR_UNIFIED_API_KEY' else 'No (Using Placeholder)'}")

if not API_KEY or API_KEY == "YOUR_UNIFIED_API_KEY":
    print("\n[ERROR] Please update .env file with your real API Key!")
    exit(1)

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Hello, are you working?"}
    ]
}

try:
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=10
    )
    
    print(f"\nStatus Code: {response.status_code}")
    if response.status_code == 200:
        print("[SUCCESS] AI Responded:")
        print(response.json()['choices'][0]['message']['content'])
    else:
        print(f"[FAILED] Response: {response.text}")
except Exception as e:
    print(f"[ERROR] Exception: {e}")
