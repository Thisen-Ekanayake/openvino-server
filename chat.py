import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("./mistral-7b-ov", "CPU")

pipe.start_chat()

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break

    def streamer(word):
        print(word, end="", flush=True)
        return False

    print("Llama: ", end="", flush=True)
    pipe.generate(
        user_input,
        max_new_tokens=300,
        streamer=streamer
    )
    print("\n")

pipe.finish_chat()