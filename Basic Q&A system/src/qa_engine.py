def answer_question(llm, context_chunks, question):
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a study assistant. 
    Answer only from the context.
    If not found, say "Not found in the document"
    
    Context:
    {context}
    
    Question:
    {question}
    """

    result = llm(prompt, max_tokens=256, temperature=0.2, top_p=0.9, stop=["Question:", "Context:"])
    return result["choices"][0]["text"].strip()