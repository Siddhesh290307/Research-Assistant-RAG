from ollama import chat

response = chat(model="llama2:latest", messages=[{"role":"user","content":"Hello"}])
print(response)