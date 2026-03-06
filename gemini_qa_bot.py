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

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 3: Text Splitter
# Language models have context limits, so we split large documents into smaller chunks.
def text_splitter(document_data):
    # We split by characters, keeping chunks up to 1000 characters with some overlap to retain context.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    # Split the document data we loaded in Step 2 into smaller list of chunks
    chunks = splitter.split_documents(document_data)
    return chunks

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Step 4: Embeddings and Vector Database
# Now we convert those text chunks into numerical vectors (embeddings) and store them in a database.
def vector_database(chunks):
    # Initialize Google's Gemini Embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create a vector database using FAISS. We pass the chunks and the embedding model.
    # It will automatically embed the chunks and store them.
    vectordb = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    return vectordb

from langchain_google_genai import ChatGoogleGenerativeAI

# Step 5: LLM and QA Chain
# Now we set up the Gemini Language Model to read the retrieved documents and answer questions.

# 5a. Set up the LLM
def get_llm():
    # Initialize Google's Gemini Model. We use gemini-1.5-pro for good reasoning.
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    return llm

# 5b. Combine the previous steps to get a "Retriever"
# A retriever is an interface that returns documents given an unstructured query.
def retriever(file_path):
    # 1. Load the document
    loaded_docs = document_loader(file_path)
    # 2. Split the text
    chunks = text_splitter(loaded_docs)
    # 3. Create the vector database
    vectordb = vector_database(chunks)
    # 4. Expose the vector database as a retriever to search for similarities
    retriever_obj = vectordb.as_retriever()
    return retriever_obj

# 5c. The QA Chain
# This step takes the user's question, searches using the retriever, and sends the prompt to the LLM.
def retriever_qa(file_path, query):
    llm = get_llm()
    retriever_obj = retriever(file_path)
    
    # 1. Retrieve relevant documents
    docs = retriever_obj.invoke(query)
    
    # 2. Combine the document content into a single string
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. Create a prompt for the Gemini LLM
    prompt = f"Use the following piece of context to answer the question. If you don't know the answer, just say you don't know.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    
    # 4. Ask the question
    response = llm.invoke(prompt)
    
    # Return the generated text
    return response.content

# ----------------------------------------------
# Create the Gradio interface
# ----------------------------------------------
def create_app():
    # We define the Gradio Interface
    # fn: The function to call when the user clicks submit
    # inputs: A File upload component for PDF and a Textbox for the question
    # outputs: A Textbox to display the answer
    app = gr.Interface(
        fn=retriever_qa,
        inputs=[
            gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
            gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
        ],
        outputs=gr.Textbox(label="Answer from Document"),
        title="Gemini QA Bot",
        description="Upload a PDF document and ask any question. The chatbot will answer using Gemini based on the provided document."
    )
    return app

if __name__ == "__main__":
    print("Welcome to the Gemini QA Bot Course!")
    
    # Check if the Gemini API Key is set
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable not found.")
        print("Please create a .env file and add your API key like this:")
        print("GEMINI_API_KEY=your_key_here")
    else:
        # Launch the Gradio app
        rag_application = create_app()
        # This will start a local web server (usually at http://127.0.0.1:7860)
        rag_application.launch(server_name="127.0.0.1")
# End of File
