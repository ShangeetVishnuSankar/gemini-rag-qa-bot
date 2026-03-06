import os
from dotenv import load_dotenv
import gradio as gr

# Load environment variables (like your Gemini API Key)
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader

# We will build our functions here step by step
# ----------------------------------------------

# Step 2: Document Loader
# This function takes the uploaded PDF file from Gradio and loads its text.
def document_loader(file_path):
    # PyPDFLoader helps us read the text inside a PDF Document
    loader = PyPDFLoader(file_path)
    loaded_document = loader.load()
    return loaded_document

# ----------------------------------------------
# Create the Gradio interface
# ----------------------------------------------
def create_app():
    # We will define the UI here in Step 6
    pass

if __name__ == "__main__":
    print("Welcome to the Gemini QA Bot Course!")
    # We will launch the app here
    
