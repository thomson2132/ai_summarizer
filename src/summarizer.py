import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from src.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

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

summarizer = SimpleSummarizer()
