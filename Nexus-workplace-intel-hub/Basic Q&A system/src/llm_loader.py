from llama_cpp import Llama

def load_llm():
    return Llama.from_pretrained(
        repo_id="TheBloke/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_ctx=4096,
        n_threads=6,
        n_batch=128,
        verbose=False,
        logits_all=False,
    )

def run_llm(llm, prompt: str, max_tokens: int = 256):
    result = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.3,
        top_p=0.9,
        stop=["</s>"]
    )
    return result["choices"][0]["text"].strip()