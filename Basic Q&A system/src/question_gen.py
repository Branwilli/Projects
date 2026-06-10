def generate_questions(llm, text):
    prompt = f"""Generate study questions from this material.
    Include recall and application questions. 
    
    Text:
    {text}
    """

    return llm(prompt, max_tokens=300)["choices"][0]["text"]