# Gemini QA Bot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about uploaded PDF documents using Google Gemini and a Gradio web interface.

## How It Works

1. Upload a PDF document
2. The document is split into chunks and embedded using Gemini
3. Your question is matched against the most relevant chunks
4. Gemini generates an answer based on that context

## Tech Stack

- **LLM**: `gemini-2.5-flash` via `langchain-google-genai`
- **Embeddings**: `gemini-embedding-001`
- **Vector Store**: FAISS
- **UI**: Gradio
- **PDF Loader**: LangChain + PyPDF

## Setup

**1. Clone the repo and create a virtual environment**
```bash
git clone https://github.com/ShangeetVishnuSankar/gemini-rag-qa-bot.git
cd gemini-rag-qa-bot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**2. Install dependencies**
```bash
pip install langchain langchain-community langchain-google-genai langchain-text-splitters faiss-cpu gradio python-dotenv pypdf
```

**3. Add your API key**

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```
Get a key at [aistudio.google.com](https://aistudio.google.com)

**4. Run**
```bash
python gemini_qa_bot.py
```

Open `http://127.0.0.1:7860` in your browser.

## Usage

1. Upload a PDF using the file input
2. Type a question about the document
3. Click **Submit** — the answer appears in the output box
