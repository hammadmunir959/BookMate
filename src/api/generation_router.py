"""
Generation API Router for Unified RAG Microservice
Handles text generation and answer production endpoints
"""

import logging
import time
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.core.config import config
from src.core.document_models import GenerationRequest, GenerationResult
from src.processors.generation.text_generator import GenerationPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for API
class GenerationRequestModel(BaseModel):
    """Request model for text generation"""
    model_config = {"protected_namespaces": ()}

    augmented_prompt: str = Field(..., description="Augmented prompt from retrieval service", min_length=1)
    generation_type: Optional[str] = Field("answer", description="Type of generation: answer, summary, explanation")
    model_provider: Optional[str] = Field(None, description="LLM provider: groq, openai, anthropic")
    model_name: Optional[str] = Field(None, description="Specific model name to use")
    temperature: Optional[float] = Field(None, description="Sampling temperature", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate", ge=1, le=4000)
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Previous conversation messages")
    citation_style: Optional[str] = Field(None, description="Citation formatting style")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")


class GenerationResponse(BaseModel):
    """Response model for generation results"""
    model_config = {"protected_namespaces": ()}

    success: bool
    answer: str
    citations: List[Dict[str, Any]]
    model_used: str
    processing_time: float
    token_count: Optional[int] = None
    confidence_score: Optional[float] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_hash: str
    created_at: str
    error_message: Optional[str] = None


@router.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequestModel):
    """Generate text response using LLM with the provided augmented prompt"""
    try:
        start_time = time.time()
        logger.info(f"🤖 Starting generation request: {len(request.augmented_prompt)} chars")

        # Create generation request object
        gen_request = GenerationRequest(
            augmented_prompt=request.augmented_prompt,
            generation_type=request.generation_type,
            model_provider=request.model_provider,
            model_name=request.model_name,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            conversation_history=request.conversation_history or [],
            citation_style=request.citation_style,
            session_id=request.session_id,
            user_id=request.user_id
        )

        # Generate response using the text generator
        text_generator = GenerationPipeline()
        result = await text_generator.generate(gen_request)

        # Format response
        response_data = {
            "success": result.success,
            "answer": result.answer,
            "citations": [
                {
                    "chunk_id": citation.chunk_id,
                    "document_id": citation.document_id,
                    "citation_text": citation.citation_text,
                    "page_number": citation.page_number,
                    "section": citation.section,
                    "paragraph_number": citation.paragraph_number
                }
                for citation in result.citations_used
            ],
            "model_used": result.model_used,
            "processing_time": result.processing_time,
            "token_count": result.token_count,
            "confidence_score": result.confidence_score,
            "session_id": result.session_id,
            "user_id": result.user_id,
            "request_hash": gen_request.request_hash,
            "created_at": result.created_at.isoformat()
        }

        if result.error_message:
            response_data["error_message"] = result.error_message

        logger.info(f"✅ Generation completed in {result.processing_time:.2f}s")
        return GenerationResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/generate/stream")
async def generate_text_stream(request: GenerationRequestModel):
    """Generate text response with streaming (future enhancement)"""
    # For now, just call the regular generate endpoint
    # In the future, this could implement server-sent events for streaming responses
    return await generate_text(request)
