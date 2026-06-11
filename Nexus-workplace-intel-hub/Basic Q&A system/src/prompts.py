def mistral_prompts(system_prompt: str, user_prompt: str) -> str:
    return f"""[INST] <<SYS>>
    You are a helpful study assistant. Answer ONLY using the provided context.
    <<SYS>>
    Context:
    {system_prompt}

    Question:
    {user_prompt} 
    [/INST]"""