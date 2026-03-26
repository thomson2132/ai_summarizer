# AI Research Paper Summarizer (RAG)

A Streamlit app that lets you upload a research paper PDF, generate a concise summary, and chat with the paper using retrieval-augmented generation (RAG).

## Features

- Upload and validate PDF files (up to 50 MB)
- Extract and clean PDF text
- Generate a short summary using Groq LLM
- Chunk and embed paper text with SentenceTransformers
- Store embeddings in ChromaDB for retrieval
- Ask follow-up questions in a chat interface grounded in paper context
- GPU-aware execution with optional CUDA + mixed precision settings

## Tech Stack

- UI: Streamlit
- LLM: Groq (via langchain-groq)
- Orchestration: LangChain
- Vector DB: ChromaDB
- Embeddings: sentence-transformers
- PDF parsing: PyPDFLoader (langchain-community)
- Runtime: Python, PyTorch

## Project Structure

```text
ai_summarizer/
  app.py                    # Main modular Streamlit app
  requirements.txt
  .gitignore
  src/
    config.py               # Environment-based config
    device_manager.py       # CPU/GPU selection and dtype settings
    embeddings.py           # Embedding model wrapper
    pdf_processor.py        # Extraction, cleaning, chunking
    chromadb_handler.py     # Index and retrieval logic
    summarizer.py           # LLM-based summarization
    rag_chain.py            # Prompt + answer generation chain
    utils.py                # Shared helpers
    init.py                 # Empty placeholder
  data/
    uploaded_pdfs/
    chromadb_storage/
  models/
```

## Prerequisites

- Python 3.10+ (recommended)
- A Groq API key
- Optional: NVIDIA GPU with CUDA for faster embedding inference

## Installation

1. Clone the repository:

```bash
git clone https://github.com/thomson2132/ai_summarizer.git
cd ai_summarizer
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a .env file in the project root.

Example:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMADB_PATH=./data/chromadb_storage
COLLECTION_NAME=research_papers
RETRIEVAL_K=3
SIMILARITY_THRESHOLD=0.5
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
USE_CUDA=true
CUDA_DEVICE=0
MIXED_PRECISION=fp16
```

Notes:
- GROQ_API_KEY is required.
- If no CUDA GPU is available, the app automatically falls back to CPU.

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit (usually http://localhost:8501).

## How It Works

1. Upload a PDF in the Streamlit UI.
2. The app extracts and cleans text from the document.
3. A short summary is generated from the extracted content.
4. The document is split into chunks and embedded.
5. Embeddings are stored in ChromaDB.
6. User questions are answered by retrieving relevant chunks and passing them to the Groq model with a constrained prompt.

## Troubleshooting

- Error: GROQ_API_KEY not found in .env file
  - Add GROQ_API_KEY to your .env file and restart the app.

- PDF appears to be scanned image / text extraction failed
  - The current pipeline expects machine-readable text PDFs.

- Slow performance on CPU
  - Use a smaller embedding model or enable CUDA if available.

- Dependency install issues
  - Upgrade pip first: pip install --upgrade pip

## Security

- Never commit .env or API keys.
- Rotate keys immediately if a key was accidentally exposed.

## Future Improvements

- OCR fallback for scanned PDFs
- Multi-document indexing and paper library
- Source citations with chunk-level references
- Better metadata extraction for title/authors/abstract

## License

Add a license file (for example MIT) if you plan to distribute this project publicly.
