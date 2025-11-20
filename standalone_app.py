"""
Research Paper Summarizer & Chat - Standalone Version
All-in-one file with complete functionality

Requirements:
pip install streamlit langchain-community langchain-groq langchain-core langchain-text-splitters chromadb sentence-transformers torch python-dotenv pypdf

Usage:
streamlit run standalone_app.py
"""

import os
import re
import logging
from pathlib import Path
from dotenv import load_dotenv

import torch
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==================== CONFIGURATION ====================
load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./data/chromadb_storage")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "research_papers")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 3))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
USE_CUDA = os.getenv("USE_CUDA", "true").lower() == "true"
CUDA_DEVICE = int(os.getenv("CUDA_DEVICE", 0))
MIXED_PRECISION = os.getenv("MIXED_PRECISION", "fp16")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DEVICE MANAGER ====================
class DeviceManager:
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.use_cuda = USE_CUDA and self.cuda_available
        self.device = self._get_device()
        self.mixed_precision = MIXED_PRECISION
        self._log_device_info()
    
    def _get_device(self):
        if self.use_cuda:
            try:
                torch.cuda.set_device(CUDA_DEVICE)
                device = torch.device(f"cuda:{CUDA_DEVICE}")
                logger.info(f"Using CUDA device: {CUDA_DEVICE}")
                return device
            except RuntimeError as e:
                logger.warning(f"Could not set CUDA device: {e}. Falling back to CPU")
                return torch.device("cpu")
        else:
            logger.info("Using CPU device")
            return torch.device("cpu")
    
    def _log_device_info(self):
        logger.info(f"CUDA Available: {self.cuda_available}")
        logger.info(f"Using CUDA: {self.use_cuda}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed Precision: {self.mixed_precision}")
        
        if self.cuda_available:
            logger.info(f"GPU Name: {torch.cuda.get_device_name(CUDA_DEVICE)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(CUDA_DEVICE).total_memory / 1e9:.2f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
    
    def empty_cache(self):
        if self.use_cuda:
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
    
    def get_device(self):
        return self.device
    
    def get_dtype(self):
        if self.mixed_precision == "fp16":
            return torch.float16
        elif self.mixed_precision == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

device_manager = DeviceManager()

# ==================== UTILITIES ====================
def create_directories():
    dirs = [
        "data/uploaded_pdfs",
        "data/chromadb_storage",
        "models"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {dir_path}")

def validate_pdf(file):
    if file is None:
        return False
    if file.type != "application/pdf":
        return False
    if file.size > 50 * 1024 * 1024:
        return False
    return True

def clean_text(text: str) -> str:
    text = " ".join(text.split())
    text = text.replace("\n", " ").replace("\r", " ")
    return text

def save_uploaded_file(uploaded_file):
    upload_dir = "data/uploaded_pdfs"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info(f"File saved: {file_path}")
    return file_path

def enable_tf32():
    if USE_CUDA:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TensorFloat-32 enabled for faster computation")

# ==================== PDF PROCESSOR ====================
def is_text_corrupted(text: str) -> bool:
    """Check if text is severely corrupted"""
    lines = text.split('\n')
    bad_lines = 0
    
    for line in lines:
        words = line.split()
        if len(words) > 0:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_count = max(word_counts.values()) if word_counts else 0
            if max_count > len(words) * 0.5:
                bad_lines += 1
    
    return bad_lines > len(lines) * 0.3

def clean_text_aggressively(text: str) -> str:
    """Aggressive cleaning for corrupted PDFs"""
    words = text.split()
    cleaned = []
    prev_word = None
    repeat_count = 0
    
    for word in words:
        if word.lower() == prev_word.lower() if prev_word else False:
            repeat_count += 1
            if repeat_count > 2:
                continue
        else:
            repeat_count = 0
        
        cleaned.append(word)
        prev_word = word
    
    text = ' '.join(cleaned)
    
    # Remove excessive colons, equals, dashes
    text = re.sub(r':+', ' ', text)
    text = re.sub(r'={3,}', ' ', text)
    text = re.sub(r'-{3,}', ' ', text)
    text = re.sub(r'[:\-=]{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF using LangChain"""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        text = " ".join([page.page_content for page in pages])
        text = clean_text(text)
        text = clean_text_aggressively(text)
        
        if is_text_corrupted(text):
            logger.warning("PDF text appears to be corrupted (scanned image?)")
            return "⚠️ PDF appears to be a scanned image. Text extraction failed."
        
        logger.info(f"Extracted {len(pages)} pages from PDF")
        logger.info(f"Text length: {len(text)} characters")
        return text
    
    except Exception as e:
        logger.error(f"Error extracting PDF: {str(e)}")
        raise

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Split text into chunks for embedding"""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_text(text)
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        raise

# ==================== EMBEDDING MODEL ====================
class EmbeddingModel:
    def __init__(self):
        try:
            self.device = device_manager.get_device()
            self.dtype = device_manager.get_dtype()
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
            logger.info(f"Embedding model device: {self.device}, dtype: {self.dtype}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def embed_text(self, text: str):
        try:
            with torch.no_grad():
                embedding = self.model.encode(
                    text,
                    convert_to_tensor=True,
                    device=self.device
                )
            return embedding.cpu().numpy()
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise
    
    def embed_batch(self, texts: list):
        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    device=self.device,
                    batch_size=32,
                    show_progress_bar=True
                )
            logger.info(f"Generated {len(embeddings)} embeddings on {self.device}")
            return embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error batch embedding: {str(e)}")
            raise
    
    def __del__(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

# ==================== CHROMADB HANDLER ====================
class ChromaDBHandler:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        try:
            self.client = chromadb.PersistentClient(path=CHROMADB_PATH)
            logger.info(f"ChromaDB initialized at {CHROMADB_PATH}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise
    
    def add_paper(self, text: str, paper_name: str):
        try:
            try:
                self.client.delete_collection(name=COLLECTION_NAME)
                logger.info("Deleted old collection")
            except:
                pass
            
            collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
            chunks = chunk_text(text)
            chunks = [c for c in chunks if len(c.strip()) > 15]
            
            if not chunks:
                logger.error("No valid chunks created from text")
                return
            
            logger.info(f"Creating embeddings for {len(chunks)} chunks...")
            embeddings = self.embedding_model.embed_batch(chunks)
            
            ids = [f"{paper_name}_{i}" for i in range(len(chunks))]
            
            collection.add(
                ids=ids,
                embeddings=embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
                metadatas=[
                    {
                        "source": paper_name,
                        "chunk_id": i,
                        "chunk_length": len(chunk.split()),
                        "is_metadata": self._is_metadata_chunk(chunk)
                    } 
                    for i, chunk in enumerate(chunks)
                ],
                documents=chunks
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB")
            
            if device_manager.device.type == "cuda":
                torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"Error adding paper to ChromaDB: {str(e)}")
            raise
    
    def _is_metadata_chunk(self, chunk: str) -> bool:
        """Check if chunk contains metadata (title, authors, abstract)"""
        metadata_keywords = ['abstract', 'keywords', 'author', 'authors', 'university', 
                            'affiliation', 'correspondence', 'received', 'accepted', 'citation']
        return any(keyword in chunk.lower() for keyword in metadata_keywords)
    
    def retrieve(self, query: str, k: int = 3):
        try:
            collection = self.client.get_collection(name=COLLECTION_NAME)
            
            logger.info(f"Querying for: {query[:60]}")
            
            # Check if this is a metadata query
            metadata_keywords = ['title', 'author', 'abstract', 'university', 'affiliation', 'email']
            is_metadata_query = any(kw in query.lower() for kw in metadata_keywords)
            
            # For metadata queries, prioritize early chunks
            if is_metadata_query:
                all_docs = collection.get()
                if all_docs and all_docs.get("documents"):
                    doc_metadata = list(zip(
                        all_docs["documents"],
                        all_docs.get("metadatas", [{}] * len(all_docs["documents"]))
                    ))
                    
                    early_chunks = []
                    other_chunks = []
                    
                    for doc, meta in doc_metadata:
                        chunk_id = meta.get("chunk_id", 999)
                        if chunk_id < 10 or meta.get("is_metadata", False):
                            early_chunks.append(doc)
                        else:
                            other_chunks.append(doc)
                    
                    docs = early_chunks[:k] + other_chunks[:max(1, k-len(early_chunks))]
                    logger.info(f"Retrieved {len(docs)} documents (metadata query mode)")
                    return docs[:k]
            
            # Regular semantic search
            query_embedding = self.embedding_model.embed_text(query)
            k_candidates = min(k * 3, 15)
            
            results = collection.query(
                query_embeddings=[query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding],
                n_results=k_candidates
            )
            
            if results["documents"] and len(results["documents"]) > 0:
                docs = results["documents"][0]
                distances = results.get("distances", [0]*len(docs))[0] if results.get("distances") else [0]*len(docs)
                metadatas = results.get("metadatas", [{}]*len(docs))[0] if results.get("metadatas") else [{}]*len(docs)
                
                scored_docs = []
                for doc, distance, metadata in zip(docs, distances, metadatas):
                    if len(doc.strip()) < 15:
                        continue
                    
                    score = distance
                    if metadata.get("is_metadata", False):
                        score *= 0.6
                    
                    scored_docs.append((doc, score))
                
                scored_docs.sort(key=lambda x: x[1])
                docs = [doc for doc, _ in scored_docs[:k]]
            else:
                docs = []
            
            logger.info(f"Retrieved {len(docs)} relevant documents")
            return docs
        
        except Exception as e:
            logger.error(f"Error retrieving from ChromaDB: {str(e)}")
            return []

# ==================== SUMMARIZER ====================
class SimpleSummarizer:
    def __init__(self):
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model_name=GROQ_MODEL,
                temperature=0.3
            )
            logger.info("Summarizer initialized")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise
    
    def summarize(self, text: str):
        try:
            text = text[:4000]
            
            if len(text.split()) < 50:
                return "⚠️ Text too short or corrupted. Please try a different PDF."
            
            prompt = PromptTemplate(
                input_variables=["text"],
                template="""Summarize the following text in 1-2 clear paragraphs.
Be concise and accurate:

Text:
{text}

Summary:"""
            )
            
            response = self.llm.invoke(prompt.format(text=text))
            summary = response.content
            
            return summary
        
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return f"Summarization failed: {str(e)}"

# ==================== RAG CHAIN ====================
class RAGChain:
    def __init__(self, chroma_handler):
        self.chroma_handler = chroma_handler
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model_name=GROQ_MODEL,
                temperature=0.3
            )
            
            template = """You are a helpful research paper assistant. Answer based on the provided context from the paper.

RULES:
1. Use information from the context provided
2. For questions about paper title, authors, abstract, or metadata: Look carefully in the context - this information is usually at the beginning
3. If you can find partial information, provide what you found
4. Only say "I couldn't find this information" if you've genuinely looked and found nothing relevant
5. Be clear and concise in your answers
6. For metadata questions (title, authors), the information is typically in the first sections of the paper

Context from paper:
{context}

Question: {question}

Answer:"""
            
            self.prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=template
            )
            
            self.chain = self.prompt | self.llm | StrOutputParser()
            logger.info("RAG chain initialized with strict fact-checking")
        
        except Exception as e:
            logger.error(f"Error initializing RAG chain: {str(e)}")
            raise
    
    def answer_question(self, question: str):
        try:
            k_results = RETRIEVAL_K
            metadata_keywords = ['author', 'title', 'abstract', 'university', 'affiliation', 'email', 'name']
            
            if any(keyword in question.lower() for keyword in metadata_keywords):
                k_results = 8
            
            context_docs = self.chroma_handler.retrieve(question, k=k_results)
            
            if not context_docs:
                return "I couldn't find relevant information in the paper for this question."
            
            valid_docs = [doc for doc in context_docs if len(doc.strip()) > 15]
            
            if not valid_docs:
                return "The retrieved content is too short to answer this question reliably."
            
            logger.info(f"Using {len(valid_docs)} documents for answer")
            for i, doc in enumerate(valid_docs):
                logger.info(f"Doc {i}: {doc[:80]}...")
            
            context = "\n\n---DOCUMENT BOUNDARY---\n\n".join(valid_docs)
            
            response = self.chain.invoke({
                "context": context,
                "question": question
            })
            
            logger.info(f"Answer: {response[:100]}")
            return response
        
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return f"Error: {str(e)}"

# ==================== STREAMLIT APP ====================
def main():
    enable_tf32()
    create_directories()
    
    logger.info(f"Running on device: {device_manager.device}")
    
    # Initialize models (cached in session state)
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = EmbeddingModel()
    if 'chroma_handler' not in st.session_state:
        st.session_state.chroma_handler = ChromaDBHandler(st.session_state.embedding_model)
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = SimpleSummarizer()
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = RAGChain(st.session_state.chroma_handler)
    
    st.set_page_config(
        page_title="Research Paper Summarizer",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "current_paper" not in st.session_state:
        st.session_state.current_paper = None
    if "paper_text" not in st.session_state:
        st.session_state.paper_text = None
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("⚙️ System Info")
        if device_manager.device.type == "cuda":
            st.success(f"✅ GPU Accelerated: {device_manager.device}")
        else:
            st.warning("⚠️ Running on CPU")
    
    # Main Title
    st.title("📚 Research Paper Summarizer & Chat")
    st.markdown("""
    Upload a research paper (PDF), get an instant AI-powered summary, 
    and chat about its content using RAG (Retrieval-Augmented Generation).

    **Powered by:** Groq LLM + ChromaDB + Sentence Transformers
    """)
    
    st.markdown("---")
    
    # SECTION 1: FILE UPLOAD
    st.header("📄 Step 1: Upload Paper")
    
    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")
    
    if uploaded_file:
        if not validate_pdf(uploaded_file):
            st.error("❌ Invalid PDF or file too large (max 50MB)")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info(f"📁 File: {uploaded_file.name}")
                st.info(f"📊 Size: {uploaded_file.size / (1024*1024):.2f} MB")
            
            st.markdown("---")
            
            # SECTION 2: SUMMARIZE BUTTON
            st.header("🤖 Step 2: Summarize Paper")
            
            if st.button("🚀 Process & Summarize", key="process_btn", use_container_width=True):
                with st.spinner("⏳ Processing paper..."):
                    try:
                        file_path = save_uploaded_file(uploaded_file)
                        
                        with st.spinner("📖 Extracting text..."):
                            paper_text = extract_pdf_text(file_path)
                        
                        with st.spinner("🤖 Generating summary..."):
                            summary = st.session_state.summarizer.summarize(paper_text)
                        
                        with st.spinner("📚 Indexing paper..."):
                            st.session_state.chroma_handler.add_paper(paper_text, uploaded_file.name)
                        
                        st.session_state.current_paper = uploaded_file.name
                        st.session_state.paper_text = paper_text
                        st.session_state.summary = summary
                        
                        st.success("✅ Paper processed successfully!")
                        st.balloons()
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        logger.error(f"Processing error: {str(e)}")
            
            # SECTION 3: DISPLAY SUMMARY
            if st.session_state.summary:
                st.markdown("---")
                st.header("📝 Summary")
                st.write(st.session_state.summary)
            
            # SECTION 4: CHAT BOX
            if st.session_state.current_paper:
                st.markdown("---")
                st.header("💬 Chat About This Paper")
                
                # Display chat history
                st.subheader("Chat History")
                chat_container = st.container(height=400)
                
                with chat_container:
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.write(message["content"])
                
                # Chat input
                user_input = st.chat_input("Ask a question about the paper...")
                
                if user_input:
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    
                    # Display user message immediately
                    with chat_container:
                        with st.chat_message("user"):
                            st.write(user_input)
                    
                    # Generate response
                    with st.spinner("🤔 Thinking..."):
                        try:
                            response = st.session_state.rag_chain.answer_question(user_input)
                            
                            # Add assistant message
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                            # Display assistant message
                            with chat_container:
                                with st.chat_message("assistant"):
                                    st.write(response)
                            
                            st.rerun()
                        
                        except Exception as e:
                            error_msg = f"❌ Error: {str(e)}"
                            st.error(error_msg)
                            logger.error(f"Chat error: {str(e)}")
                
                # Clear chat button
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()
    else:
        st.info("👆 Upload a PDF file to get started!")
        
        # Show example use cases
        st.markdown("""
        ### 📚 How it works:
        1. **Upload** - Select a research paper (PDF format)
        2. **Summarize** - Click to generate AI summary
        3. **Chat** - Ask questions about the paper with RAG

        ### 🎯 Supported papers:
        - ArXiv papers
        - PubMed/biomedical papers
        - Conference papers
        - Journal articles
        
        ### ⚡ Features:
        - GPU-accelerated processing (CUDA)
        - Handles papers up to 50MB
        - High-quality summarization
        - Context-aware Q&A with RAG
        """)

if __name__ == "__main__":
    main()
