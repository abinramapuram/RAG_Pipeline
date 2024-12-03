
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import re
import fitz
import pandas as pd


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


# Extract text from Chapter 2 and 3
chapter_2_text = extract_text_from_pdf("ConceptsofBiology-WEB.pdf", 40, 101)


# Chunk text
chunks = chunk_text(chapter_2_text)

# Generate embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(chunks, convert_to_numpy=True)

# Create FAISS index
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# Save mappings of chunks
chunk_mapping = {i: chunk for i, chunk in enumerate(chunks)}

# Load HuggingFace model
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")


# Load the test data CSV file
file_path = "Test_data.csv"  # Adjust the path to where your CSV file is stored
data = pd.read_csv(file_path)


outputs = []
# Iterating over all the queries in the csv.
for query in data['Question']:
    
    try:
        context = retrieve_relevant_chunks(query)
        
        prompt = f"Context: {context}\n\nQuestion: {query}"
        
        response = qa_model(prompt, max_new_tokens=500, num_return_sequences=1)
        
        answer = response[0]['generated_text']
        
        print('\n', answer)
        
    except Exception as e:
        answer = f"Error: {e}"
    
    outputs.append(answer)

output_column='Output'
# Add the outputs to the DataFrame
data[output_column] = outputs


# Save the updated DataFrame back to a file
updated_file_path = "Test_data_with_outputs.csv"  # Specify your desired output path
data.to_csv(updated_file_path, index=False)


print(f"Updated file saved at: {updated_file_path}")

