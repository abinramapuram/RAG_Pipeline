# ------------------------ RAG_Pipeline --------------------------


## Contents

- **ConceptsofBiology-WEB.pdf**: Input PDF.
- **requirements.txt**: Python libraries required.
- **rag_pipeline.py**: RAG pipeline implementation code with single query.
- **rag_pipeline_testing.py**: RAG pipeline testing code using a set of test queries in a CSV file.
- **Test_data.csv**: Test queries and their expected answers.
- **Test_data_with_outputs.csv**: Test queries and their outputs from the RAG pipeline.
- **rag_pipeline_fastapi.py**: Implementation using FastAPI.
- **rag_pipeline_streamlit.py**: Implementation with a Streamlit frontend.
- **rag-app**: Docker implementation files.
- **RAG_Pipeline.docx**: RAG pipeline architecture details.


Setup Instructions
 
	- Install Required Libraries using the requirements.txt file:
		pip install -r requirements.txt
	- Prepare the PDF
		Place the PDF file (ConceptsofBiology-WEB.pdf) in the same directory as the script.

Run the implementation code

	Update the user query in 'query' variable.
	Run : python rag_pipeline.py

Run the testing code

	Update the test data filename in 'file_path' variable.
	Run : python rag_pipeline_testing.py


Run the streamlit app

	streamlit run rag_pipeline_streamlit.py
	
	Then open the app in your browser (http://localhost:8501).


Run as docker

	Navigate to the project directory
		cd rag-app

	Build the Docker image
		docker build -t rag-app .

	Run the Docker Container
		docker run -p 8501:8501 rag-app
		
	The app will be available at http://localhost:8501.


Run the Fastapi application

	Execute the script:
		python rag_pipeline_fastapi.py
		
	The FastAPI application will run on http://127.0.0.1:8000

	For testing open http://127.0.0.1:8000/docs in the browser.
	Post queries there and view the responses.


