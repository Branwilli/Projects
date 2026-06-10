from pathlib import Path 
from llm_loader import load_llm, run_llm
from prompts import mistral_prompts

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

print("Model exists:", MODEL_PATH.exists())

llm = load_llm(str(MODEL_PATH))

system_prompt = "You are a helpful offline study assistant."
user_prompt = "Explain what information systems are in simple terms."

prompt = mistral_prompts(system_prompt, user_prompt)

response = run_llm(llm, prompt)
print("\nResponse:\n", response)