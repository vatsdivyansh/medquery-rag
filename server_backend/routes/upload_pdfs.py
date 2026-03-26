from fastapi import APIRouter , UploadFile , File
from typing import List 
from modules.load_vectorstore import load_vectorstore # created inside the modules folder
from fastapi.responses import JSONResponse
from logger import logger 

router = APIRouter()

@router.post("/upload_pdfs/")

async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        logger.info("Received uploaded files")

       
        if not files:
            return JSONResponse(
                status_code=400,
                content={"error": "No files uploaded"}
            )

        
        load_vectorstore(files)

        logger.info("Documents added to vectorstore")

        return {
            "message": f"{len(files)} file(s) processed successfully"
        }

    except Exception as e:
        logger.exception("Error during PDF upload")

        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

