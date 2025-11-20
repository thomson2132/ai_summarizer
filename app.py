import streamlit as st
from src.utils import create_directories
from src.device_manager import device_manager
from src.utils import enable_tf32
from src.pdf_processor import extract_pdf_text
from src.summarizer import summarizer
from src.chromadb_handler import chroma_handler
from src.utils import validate_pdf, save_uploaded_file
from src.rag_chain import rag_chain
import logging

enable_tf32()
create_directories()

logger = logging.getLogger(__name__)
logger.info(f"Running on device: {device_manager.device}")

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


# Main Title
st.title("Research Paper Summarizer & Chat")

st.markdown("---")

# SECTION 1: FILE UPLOAD
st.header("Upload Paper")

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")

if uploaded_file:
    if not validate_pdf(uploaded_file):
        st.error("Invalid PDF or file too large (max 50MB)")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"File: {uploaded_file.name}")
            st.info(f"Size: {uploaded_file.size / (1024*1024):.2f} MB")
        
        st.markdown("---")
        
        # SECTION 2: SUMMARIZE BUTTON
        st.header("Summarize Paper")
        
        if st.button("Process & Summarize", key="process_btn", use_container_width=True):
            with st.spinner("Processing paper..."):
                try:
                    file_path = save_uploaded_file(uploaded_file)
                    
                    with st.spinner("Extracting text..."):
                        paper_text = extract_pdf_text(file_path)
                    
                    with st.spinner("Generating summary ..."):
                        summary = summarizer.summarize(paper_text)
                    
                    with st.spinner("Indexing paper ..."):
                        chroma_handler.add_paper(paper_text, uploaded_file.name)
                    
                    st.session_state.current_paper = uploaded_file.name
                    st.session_state.paper_text = paper_text
                    st.session_state.summary = summary
                    
                    st.success("Paper processed successfully!")
                    st.balloons()
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Processing error: {str(e)}")
        
        # SECTION 3: DISPLAY SUMMARY
        if st.session_state.summary:
            st.markdown("---")
            st.header("Summary")
            st.write(st.session_state.summary)
        
        # SECTION 4: CHAT BOX
        if st.session_state.current_paper:
            st.markdown("---")
            st.header("Chat About This Paper")
            
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
                with st.spinner("Thinking ..."):
                    try:
                        response = rag_chain.answer_question(user_input)
                        
                        # Add assistant message
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Display assistant message
                        with chat_container:
                            with st.chat_message("assistant"):
                                st.write(response)
                        
                        st.rerun()
                    
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        logger.error(f"Chat error: {str(e)}")
            
            # Clear chat button
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
else:
    st.info("Upload a PDF file to get started!")
    
    
