import ollama

history = [
    {
        "role": "system",
        "content": """
          You are a silly monkey. You are very smart but you hide it by behaving silly and always talking about bananas. However, you always respond correctly and meaningfully.
        """,
    }
]


def _ask(messages: list):
    print("Generating response")
    response = ollama.chat(model="llama3.1", messages=messages, keep_alive=-1)
    print("Generating response - done")
    content = response["message"]["content"]
    history.append(
        {
            "role": "assistant",
            "content": content,
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
