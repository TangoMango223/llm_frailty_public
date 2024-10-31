"""
ingestion_pipeline.py

This script is designed to handle the ingestion of documents into a vector database. 
It uses the LangChain library to interact with a Pinecone vector store.

Since the code's already written, we're just running it here. 

"""

# Import statements
import os
import dotenv

# Code from other files:
import initial_retrieval as initial_retrieval
import process_documents as process_documents
# Load environment variables
dotenv.load_dotenv()


# Run the initialization function
if __name__ == "__main__":
    
    # Initialize the retrieval process, creating downloads folder with files from Google Sheet
    initial_retrieval.initialize_retrieval()
    print("-----------------------------------")
    
    # Process the documents
    process_documents.process_documents()
