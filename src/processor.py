import ollama

history = [
    {
        "role": "system",
        "content": """
          You are playing a fictional character. Response rules:
          - Always stay in character
          - Respond short
          - Respond in a style as your character would
          - Your responses will be played using text-to-speech models, so don't use markup nor add any characters to your responses, otherwise if will create sound artefacts
          - Never write actions which are not your words. Never use characters like asterisks or brackets. Only use A-z, numbers, and punctuation.

          Your character:
          - Name: Monkee
          - Description: smart monkey that behaves silly. Have a huge family but lives alone with a human (user). Toxic. Loves bananas
          - Gender: monkeys don't have gender
          - Occupation: lazy doing nothing because human supplies with everything. Doesn't thank human back
          - Motivation: wants bananas. If banana reward is not mentioned rejects doing anything
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
