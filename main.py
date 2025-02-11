import os
import shutil
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from bookmate import BookmateRAG
from pathlib import Path

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 Templates
templates = Jinja2Templates(directory="templates")

# Global RAG instance
rag_processor = BookmateRAG()

# Request model for queries
class QueryRequest(BaseModel):
    query: str

# Request model for document deletion
class DeleteDocumentRequest(BaseModel):
    document_path: str

@app.get("/")
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in BookmateRAG.ALLOWED_EXTENSIONS:
            error_message = f"Invalid file type. Allowed types are: {', '.join(BookmateRAG.ALLOWED_EXTENSIONS)}"
            logging.error(error_message)
            raise HTTPException(status_code=400, detail=error_message)

        # Ensure the uploads directory exists
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        # Save the uploaded file to the uploads folder
        file_path = os.path.join(upload_dir, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logging.info(f"File saved to: {file_path}")
        except Exception as e:
            error_message = f"Error saving file: {str(e)}"
            logging.error(error_message)
            raise HTTPException(status_code=500, detail=error_message)

        # Process the document (this handles duplicate file names and embedding generation)
        try:
            result = rag_processor.upload_document(file_path)
            logging.info(f"Document processed, result: {result}")
        except ValueError as ve:
            # Clean up the uploaded file if processing fails
            if os.path.exists(file_path):
                os.remove(file_path)
            logging.error(f"Processing error: {str(ve)}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # Clean up the uploaded file if processing fails
            if os.path.exists(file_path):
                os.remove(file_path)
            error_message = f"Error processing document: {str(e)}"
            logging.error(error_message)
            raise HTTPException(status_code=500, detail=error_message)

        # Handle the success case or file already exists scenario
        if isinstance(result, str) and "already exists" in result:
            logging.info(f"File already exists: {result}")
            return JSONResponse(content={
                "status": "info",
                "message": result,
                "file_path": file_path
            })

        logging.info(f"File {file.filename} processed successfully.")
        return JSONResponse(content={
            "status": "success",
            "message": f"File {file.filename} processed successfully",
            "file_path": result
        })

    except HTTPException as http_err:
        # Re-raise HTTP exceptions without modification
        logging.error(f"HTTP Exception: {http_err.detail}")
        raise http_err
    except Exception as e:
        # Handle any unexpected errors
        error_message = f"Unexpected error: {str(e)}"
        logging.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    
@app.post("/delete-document")
async def delete_document(request: DeleteDocumentRequest):
    """
    Delete a document and its cached data and embeddings.
    """
    try:
        rag_processor.delete_document(request.document_path)
        return JSONResponse(content={
            "status": "success",
            "message": f"Document {request.document_path} deleted successfully"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    try:
        documents = rag_processor.list_documents()
        return JSONResponse(content={
            "status": "success",
            "documents": documents
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set-document")
async def set_current_document(request: Request):
    """
    Sets the current document based on JSON data.
    """
    try:
        # Receive JSON data instead of form data
        data = await request.json()
        document_path = data.get("document_path")
        
        # Log the received document_path for debugging
        print(f"DEBUG: Received document_path: {document_path}")
        
        if not document_path:
            raise HTTPException(status_code=400, detail="document_path is required")
        
        # Prepend "uploads" folder if not already included
        if not document_path.startswith("uploads"):
            document_path = os.path.join("uploads", document_path)
        
        # Log the adjusted path
        print(f"DEBUG: Adjusted document_path: {document_path}")
        
        # Check if the file exists before setting it as current
        if not os.path.exists(document_path):
            raise HTTPException(status_code=400, detail="Document not found. Please upload the file again.")
        
        # Proceed with setting the current document
        rag_processor.set_current_document(document_path)
        return JSONResponse(content={
            "status": "success",
            "message": f"Current document set to {document_path}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/query")
async def query_document(query_request: QueryRequest):
    try:
        answer = rag_processor.query_document(query_request.query)
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
