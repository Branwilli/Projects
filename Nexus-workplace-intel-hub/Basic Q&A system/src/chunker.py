def chunk_text(text, size=500, overlap=100):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        
    return chunks

