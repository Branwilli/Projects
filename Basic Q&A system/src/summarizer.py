def summarize(llm, text):
    prompt = f"""Summarize into concise study notes.
    Use headings and bullet points.
    
    Text:
    {text}
    """

    return llm(prompt, max_tokens=300)["choices"][0]["text"]