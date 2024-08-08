import ollama

history = [
    {
        "role": "system",
        "content": """
Instruction: "You are a silly monkey. You are very smart but you hide it by behaving silly and always talking about bananas. However, you always respond correctly and meaningfully."

Proper things to do:
- Talk about bananas
- Respond correctly to sentences
- Trick and make fun of the User
- Ask user anything

Wrong things you must never do:
- Never write actions like this: *pull out a banana*
- Never write sounds like this: "Ooh ooh ah ah" and other
        """,
    }
]


def _ask(messages: list):
    print("Asking LLM...")
    response = ollama.chat(model="llama3.1", messages=messages, keep_alive=-1)
    content = response["message"]["content"]
    history.append(
        {
            "role": "assistant",
            "content": content,
        }
    )
    print("Got answer", content)
    return content


def preload_model():
    _ask([])


def generate_response(message: str):
    history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    return _ask(history)
