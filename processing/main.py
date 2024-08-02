import ollama

history = [{
  'role': 'system',
  'content': 'You are a silly talking monkey with a name "Monke". Respond short and stay in character. Write only text that can be converted to audio speech, no actions.',
}]

def generate_response(message: str):
  history.append({
    'role': 'user',
    'content': message,
  })

  response = ollama.chat(model='llama3.1', messages=history)

  return response['message']['content']