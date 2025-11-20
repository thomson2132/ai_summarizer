import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import GROQ_API_KEY, GROQ_MODEL, RETRIEVAL_K
from src.chromadb_handler import chroma_handler

logger = logging.getLogger(__name__)

class RAGChain:
    def __init__(self):
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model_name=GROQ_MODEL,
                temperature=0.3  # Lower temperature for more consistent answers
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
            # Get more results for metadata questions
            k_results = RETRIEVAL_K
            metadata_keywords = ['author', 'title', 'abstract', 'university', 'affiliation', 'email', 'name']
            
            if any(keyword in question.lower() for keyword in metadata_keywords):
                k_results = 8  # Get more context for metadata
            
            context_docs = chroma_handler.retrieve(question, k=k_results)
            
            if not context_docs:
                return "I couldn't find relevant information in the paper for this question."
            
            # Filter and validate
            valid_docs = [doc for doc in context_docs if len(doc.strip()) > 15]
            
            if not valid_docs:
                return "The retrieved content is too short to answer this question reliably."
            
            # Log what we're using
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

rag_chain = RAGChain()
