import os

import requests
from dotenv import load_dotenv

load_dotenv()


def main():
    api_key = os.getenv("HORDE_API_KEY")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    # Fetch available models
    models = requests.get(
        "http://127.0.0.1:8000/v1/chat/models",
        headers=headers,
        params={"base_models": "llama-3-instruct"},
    )
    model_names = [model["name"] for model in models.json()]

    print(f"Models: {model_names}")

    # Get completions
    response = requests.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        headers=headers,
        json={
            "model": ",".join(model_names),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        },
    )
    print(response.json())


if __name__ == "__main__":
    main()
