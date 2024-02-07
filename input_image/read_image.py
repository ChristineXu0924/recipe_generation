import base64
import requests
import json
from openai import OpenAI

# masking the api key just for security reasons
# api_key = "sk-LXA0*****************************DCQHpz"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        # print(type(image_file))
        return base64.b64encode(image_file.read()).decode('utf-8')

def ingredient_identifier(image_path):
    base64_image = encode_image(image_path)
    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }
    payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "Please identify all the food ingredients in the image, response in json format with the list of items."
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 300
            }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

