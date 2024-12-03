# Use an official Python image as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the application code and requirements into the container
COPY rag_pipeline_streamlit.py /app/rag_pipeline_streamlit.py
COPY requirements.txt /app/requirements.txt
COPY ConceptsofBiology-WEB.pdf /app/ConceptsofBiology-WEB.pdf

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Expose the port Streamlit uses
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "rag_pipeline_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]

