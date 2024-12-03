
import fitz  # PyMuPDF for text extraction
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Request model
class Query(BaseModel):
    question: str

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path, start_page, end_page):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(start_page, end_page + 1):
            page = pdf[page_num]
            text += page.get_text()
    return text

# Function to chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, k)
    relevant_chunks = [chunk_mapping[idx] for idx in indices[0]]
    return " ".join(relevant_chunks)

# Setup RAG pipeline
def setup_rag_pipeline():
    pdf_path = "ConceptsofBiology-WEB.pdf"  
    chapter_2_text = extract_text_from_pdf(pdf_path, 40, 101)
    chunks = chunk_text(chapter_2_text)

    # Generate embeddings
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_numpy=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    # Map chunks
    chunk_mapping = {i: chunk for i, chunk in enumerate(chunks)}

    # Load QA model
    qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

    return model, faiss_index, chunk_mapping, qa_model

# Initialize RAG components
model, faiss_index, chunk_mapping, qa_model = setup_rag_pipeline()

# API endpoint to process queries
@app.post("/ask")
async def ask_question(query: Query):
    try:
        # Retrieve context
        context = retrieve_relevant_chunks(query.question)

        # Generate response
        prompt = f"Context: {context}\n\nQuestion: {query.question}"

        response = qa_model(prompt, max_new_tokens=500, num_return_sequences=1)
        answer = response[0]['generated_text']

        return {"question": query.question, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app (only if running directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
