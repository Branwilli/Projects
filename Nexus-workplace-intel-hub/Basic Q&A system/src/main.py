from extractor import extract_document
from chunker import chunk_text
from embedder import Embedder
from question_gen import generate_questions
from store import retrieve
from llm_loader import load_llm, run_llm
from qa_engine import answer_question
from prompts import mistral_prompts

from pathlib import Path

from summarizer import summarize

BASE_DIR = Path(__file__).resolve().parents[1]  # project root
DOC_PATH = BASE_DIR / "data" / "uploads" / "CaseStudy.pdf"

doc = extract_document(str(DOC_PATH))

chunks = chunk_text(doc['sections'])
chunks = [
    chunk["text"]
    for sublist in chunks
    for chunk in sublist
    if isinstance(chunk, dict)
]

embedder = Embedder()
embeddings = embedder.embed(chunks)

llm = load_llm()

# --- Mode selection ---
print("\nSelect mode:")
print("1 - Answer a question")
print("2 - Summarize document")
print("3 - Generate study questions")
mode = input("\nEnter choice (1/2/3): ").strip()

if mode == "1":
    user_prompt = input("Enter your question: ").strip()
    q_vec = embedder.embed([user_prompt])[0]
    top_chunks = retrieve(q_vec, embeddings, chunks, top_k=3)
    answer = answer_question(llm, top_chunks, user_prompt)
    print("\nANSWER:\n", answer)

elif mode == "2":
    # Use the full document text for summarization
    full_text = "\n\n".join(chunks)
    summary = summarize(llm, full_text)
    print("\nSUMMARY:\n", summary)

elif mode == "3":
    # Use the full document text for question generation
    full_text = "\n\n".join(chunks)
    questions = generate_questions(llm, full_text)
    print("\nSTUDY QUESTIONS:\n", questions)

else:
    print("Invalid choice. Please enter 1, 2, or 3.")
