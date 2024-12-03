
import fitz  # PyMuPDF for text extraction
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import streamlit as st
import time

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

def normalize_query(query):
    """
    Ensures the query ends with a '?' if it's not already punctuated.
    """
    query = query.strip()
    if not query.endswith("?"):
        query += "?"
    return query

# Streamlit app
st.title("RAG Application for Concepts of Biology")
st.write("Enter your query from the concepts of biology")

# Input for query
user_query = st.text_input("Enter your question:")

# Pre-load RAG components
@st.cache_resource
def setup_rag_pipeline():
    # Load and preprocess the text
    pdf_path = "ConceptsofBiology-WEB.pdf"
    chapter_2_text = extract_text_from_pdf(pdf_path, 40, 101)

    # Chunk text
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

# Set up the RAG pipeline
model, faiss_index, chunk_mapping, qa_model = setup_rag_pipeline()

# Display action
if user_query:
    
    # Adding '?' to query if it not there
    user_query = normalize_query(user_query)
    print(user_query)

    # Retrieve relevant chunks
    context = retrieve_relevant_chunks(user_query)
    # print(context)

    # Prompt creation
    prompt = f"Context: {context}\n\nQuestion: {user_query}"

    # Generate response
    response = qa_model(prompt, max_new_tokens=500, num_return_sequences=1)
    answer = response[0]['generated_text']
    
    print(answer)

    # Display results
    st.write("### Response:")
    st.write(answer)
