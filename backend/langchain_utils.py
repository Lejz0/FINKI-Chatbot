from langchain_groq import ChatGroq

from config import settings

def get_llm():
    llm = ChatGroq(
          model_name="llama-3.3-70b-versatile", 
               temperature=0, 
               groq_api_key=settings.groq_api_key, 
               streaming=True)
    
    return llm

