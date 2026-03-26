from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from pydantic import Field
from typing import List, Optional
from logger import logger
import os
from dotenv import load_dotenv
from langchain_core.retrievers import BaseRetriever

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")       
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")   
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  

router = APIRouter()

@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"User query:{question}")

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)          

        embed_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",         
            google_api_key=GOOGLE_API_KEY
        )
        embedded_query = embed_model.embed_query(question)
        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)

        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            ) for match in res["matches"]
        ]

        class SimpleRetriever(BaseRetriever):
            tags: Optional[List[str]] = Field(default_factory=list)    
            metadata: Optional[dict] = Field(default_factory=dict)     
            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs

        retriever = SimpleRetriever(docs)
        chain = get_llm_chain(retriever=retriever)
        result = query_chain(chain, question)
        logger.info("Query is successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})