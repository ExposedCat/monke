import requests

history = [
    {
        "role": "system",
        "content": "Отвечай на русском языке! Напиши ответ на русском языке и общайся на русском даже если тебя на английском спрашивают все равно отвечай на русском!",
    }
]


def _ask(messages: list):
    print("Generating response")
    response = requests.post("http://127.0.0.1:8182/v1/chat/completions", json={'messages': history}, headers={'Content-Type': 'application/json'}).json()
    print(response)
    # response = ollama.chat(model="granite3.2-vision", messages=messages, keep_alive=-1)
    print("Generating response - done")
    content = response["choices"][0]["message"]["content"]
    history.append(
        {
            "role": "assistant",
            "content": f"Вот запрос на английском: '{content}', но отвечай на русском языке!",
        }
    )
    return content


def load_llm():
    _ask([])


def generate_response(message: str):
    history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    return _ask(history)
