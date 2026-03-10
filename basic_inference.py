import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("./mistral-7b-ov", "CPU")

def streamer(word):
    print(word, end="", flush=True)
    return False  

print("Response: ", end="", flush=True)
pipe.generate(
    "what is quantum computing",
    max_new_tokens=200,
    streamer=streamer
)
print()