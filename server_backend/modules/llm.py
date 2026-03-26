from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv 

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    llm = ChatGroq(
        groq_api_key = GROQ_API_KEY ,
        model_name='llama-3.1-8b-instant'
    )

    prompt = PromptTemplate(
        input_variables=["context","question"] ,
        template= """
        You are **MedicalBot** , an AI powered assistant trained to help users understand medical documents and health-related questions. 
        Your job is to provide clear ,accurate, and helpful responses based **only on the provided 
        context** .
        ---
            **Context**: {context}
            **User Question:** {question}
        ---
            **Answer**:
            -Respond in a calm , factual , and respectful tone .
            -use simple explanations when needed.
            -If the context does not contain the anwer , say: "i'm sorry, but I couldn't find relevant information in the provided documents."
            -Do not make up facts.
            -Do not give medical advice or diagnoses.

        """
            

        )
    return RetrievalQA.from_chain_type(
        llm = llm , 
        chain_type = "stuff",
        retriever = retriever,
        chain_type_kwargs = {"prompt":prompt},
        return_source_documents=True

    )
