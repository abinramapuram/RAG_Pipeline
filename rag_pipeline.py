from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import fitz
import time


def extract_text_from_pdf(pdf_path, start_page, end_page):
    """
    Extract text from specific pages of a PDF.
    """
    text = ""

    # Open the PDF
    with fitz.open(pdf_path) as pdf:
        for page_num in range(start_page, end_page + 1):
            page = pdf[page_num]
            text += page.get_text()  # Extract text from the page
    return text


def chunk_text(text, chunk_size=500):
    """
    Split text into smaller chunks of specified size.
    """
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def retrieve_relevant_chunks(query, k=3):
    """
    Retrieve top-k relevant text chunks for a given query.
    """
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, k)
    relevant_chunks = [chunk_mapping[idx] for idx in indices[0]]
    return " ".join(relevant_chunks)

def normalize_query(query):
    """
    Ensures the query ends with a '?' if it's not already punctuated.
    """
    query = query.strip()
    if not query.endswith("?"):
        query += "?"
    return query


start_time = time.time()

pdf_input = "ConceptsofBiology-WEB.pdf"                      # input data
embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'   # Embedding model for encoding of query and data
model_llm = "google/flan-t5-base"                            # LLM model for generation

# Enter the query from the user
query = "What are the differences between prokaryotic and eukaryotic cells?"

# Adding '?' to query if it not there
query = normalize_query(query)                              

# Extract text from Chapter 2 and 3
chapter_2_text = extract_text_from_pdf(pdf_input, 40, 101)


# Chunking text
chunks = chunk_text(chapter_2_text)

# Generate embeddings
model = SentenceTransformer(embedding_model)
embeddings = model.encode(chunks, convert_to_numpy=True)

# Create FAISS index
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# Save mappings of chunks
chunk_mapping = {i: chunk for i, chunk in enumerate(chunks)}

# Load HuggingFace model
qa_model = pipeline("text2text-generation", model= model_llm)

# Retrieved context
context = retrieve_relevant_chunks(query)

# Prompt creation
prompt = f"Context: {context}\n\nQuestion: {query}"

# LLM inference
response = qa_model(prompt, max_new_tokens=500, num_return_sequences=1)

# Generated output
answer = response[0]['generated_text']

print(answer)

print("--- %s time in seconds ---" % (time.time() - start_time))
