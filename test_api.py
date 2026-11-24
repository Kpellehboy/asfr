
import requests
import json

API_URL = "http://127.0.0.1:8000/predict"
IMAGE_PATH = "test_images/20231006_164208.jpg"

print("Sending image to FastAPI...")

with open(IMAGE_PATH, "rb") as image:
    response = requests.post(
        API_URL,
        files={
            "file": (
                "20231006_164208.jpg",
                image,
                "image/jpeg"
            )
        },
        timeout=120
    )

print("\nHTTP STATUS:", response.status_code)

try:
    result = response.json()
    print("\nAPI RESPONSE:")
    print(json.dumps(result, indent=4))

except ValueError:
    print(response.text)

response.raise_for_status()

print("\nAPI PREDICTION TEST PASSED")
