import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import re

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import hashlib
from rank_bm25 import BM25Okapi
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/logs/rag.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
api = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation service with domain weighting",
    version="1.0.0"
)

# Models
class SearchRequest(BaseModel):
    query: str
    domain_weights: Dict[str, float] = {}
    k: int = 5
    threshold: float = 0.7
    use_bm25_reranking: bool = True
    bm25_weight: float = 0.3
    embedding_weight: float = 0.7

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    evidence: str
    search_time: float
    reranking_method: Optional[str] = None

class DocumentRequest(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}
    domain: str = "general"
    source: Optional[str] = None
    source_type: Optional[str] = "text"  # text, asr, file
    timestamps: Optional[List[Dict[str, Any]]] = None  # For ASR content with timestamps
    citations: Optional[List[Dict[str, Any]]] = None  # Pre-existing citations

class DocumentResponse(BaseModel):
    document_id: str
    chunks_created: int
    embedding_dimension: int

class RAGService:
    """Main RAG service managing embeddings and retrieval"""
    
    def __init__(self):
        self.embedding_model = None
        self.qdrant_client = None
        self.collection_name = "documents"
        self.embedding_dimension = 384  # for all-MiniLM-L6-v2
        
        # BM25 index for keyword-based retrieval
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_documents = []
        
        # Domain mappings
        self.domain_mapping = {
            "chemistry": ["chemical", "molecule", "reaction", "compound", "element"],
            "mechanical": ["force", "stress", "strain", "material", "engineering"],
            "materials": ["steel", "aluminum", "composite", "alloy", "properties"]
        }
    
    async def initialize(self):
        """Initialize embedding model and Qdrant connection"""
        try:
            # Download NLTK data
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
            except:
                logger.warning("Could not download NLTK data, BM25 may be less effective")
            
            # Initialize embedding model
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer(
                os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            )
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Dimension: {self.embedding_dimension}")
            
            # Initialize Qdrant client
            qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
            logger.info(f"Connecting to Qdrant at {qdrant_url}")
            self.qdrant_client = QdrantClient(url=qdrant_url)
            
            # Create collection if it doesn't exist
            await self.ensure_collection_exists()
            
            # Build BM25 index from existing documents
            await self.build_bm25_index()
            
            logger.info("RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise
    
    async def ensure_collection_exists(self):
        """Ensure the document collection exists in Qdrant"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing"""
        try:
            # Convert to lowercase and remove special characters
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
            # Split into tokens
            try:
                from nltk.tokenize import word_tokenize
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words('english'))
                tokens = word_tokenize(text)
                # Remove stopwords and short tokens
                tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
            except:
                # Fallback if NLTK is not available
                tokens = [word for word in text.split() if len(word) > 2]
            
            return tokens
        except Exception as e:
            logger.warning(f"Error preprocessing text: {e}")
            return text.split()
    
    async def build_bm25_index(self):
        """Build BM25 index from existing documents in Qdrant"""
        try:
            if not self.qdrant_client:
                return
            
            # Get all documents from Qdrant
            try:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=10000  # Adjust based on your dataset size
                )
                
                self.bm25_corpus = []
                self.bm25_documents = []
                
                for point in scroll_result[0]:
                    content = point.payload.get("content", "")
                    if content:
                        tokens = self.preprocess_text(content)
                        self.bm25_corpus.append(tokens)
                        self.bm25_documents.append({
                            "id": point.id,
                            "content": content,
                            "payload": point.payload
                        })
                
                # Build BM25 index
                if self.bm25_corpus:
                    self.bm25_index = BM25Okapi(self.bm25_corpus)
                    logger.info(f"Built BM25 index with {len(self.bm25_corpus)} documents")
                else:
                    logger.info("No documents found for BM25 index")
                    
            except Exception as e:
                logger.warning(f"Could not build BM25 index: {e}")
                
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Chunk text into smaller pieces for better retrieval"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            
            chunk = text[start:end]
            
            # Try to break at word boundaries
            if end < len(text) and not text[end].isspace():
                last_space = chunk.rfind(' ')
                if last_space > start + chunk_size // 2:
                    end = start + last_space
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def calculate_domain_score(self, text: str, domain_weights: Dict[str, float]) -> float:
        """Calculate domain relevance score for text"""
        if not domain_weights:
            return 1.0
        
        text_lower = text.lower()
        total_score = 0.0
        total_weight = 0.0
        
        for domain, weight in domain_weights.items():
            if weight <= 0:
                continue
                
            domain_keywords = self.domain_mapping.get(domain, [])
            domain_score = 0.0
            
            for keyword in domain_keywords:
                if keyword in text_lower:
                    domain_score += 0.1
            
            # Normalize domain score
            domain_score = min(domain_score, 1.0)
            total_score += domain_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def generate_citations(self, content: str, source: str, source_type: str = "text", 
                          timestamps: Optional[List[Dict[str, Any]]] = None, 
                          citations: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Generate citation metadata for content chunks"""
        generated_citations = []
        
        if citations:
            # Use pre-existing citations
            return citations
        
        if source_type == "asr" and timestamps:
            # Generate citations for ASR content with timestamps
            for timestamp_info in timestamps:
                citation = {
                    "source": source,
                    "source_type": "asr",
                    "timestamp_start": timestamp_info.get("start", 0.0),
                    "timestamp_end": timestamp_info.get("end", 0.0),
                    "segment_text": timestamp_info.get("text", ""),
                    "confidence": timestamp_info.get("confidence", 0.0),
                    "is_technical": timestamp_info.get("is_technical", False),
                    "technical_score": timestamp_info.get("technical_score", 0.0),
                    "created_at": datetime.utcnow().isoformat()
                }
                generated_citations.append(citation)
        else:
            # Generate basic citation for non-ASR content
            citation = {
                "source": source or "unknown",
                "source_type": source_type,
                "content_length": len(content),
                "created_at": datetime.utcnow().isoformat()
            }
            generated_citations.append(citation)
        
        return generated_citations
    
    def map_chunk_to_citation(self, chunk: str, chunk_index: int, citations: List[Dict[str, Any]], 
                             chunk_start: int, chunk_end: int) -> Optional[Dict[str, Any]]:
        """Map a text chunk to its corresponding citation"""
        if not citations:
            return None
        
        # For ASR content, find the citation that best matches the chunk position
        for citation in citations:
            if citation.get("source_type") == "asr":
                # Calculate overlap between chunk position and timestamp
                timestamp_start = citation.get("timestamp_start", 0.0)
                timestamp_end = citation.get("timestamp_end", 0.0)
                
                # For now, use a simple mapping based on chunk order
                # In a more sophisticated implementation, you'd map character positions to timestamps
                if chunk_index < len(citations):
                    return citation
            else:
                # For non-ASR content, return the main citation
                return citation
        
        # Return first citation as fallback
        return citations[0] if citations else None
    
    async def add_document(self, content: str, metadata: Dict[str, Any], domain: str = "general", 
                          source: Optional[str] = None, source_type: str = "text",
                          timestamps: Optional[List[Dict[str, Any]]] = None,
                          citations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Add a document to the vector database"""
        try:
            # Generate citations for the content
            document_citations = self.generate_citations(
                content, source or "unknown", source_type, timestamps, citations
            )
            
            # Chunk the document
            chunks = self.chunk_text(content)
            logger.info(f"Document chunked into {len(chunks)} pieces")
            
            # Generate embeddings for chunks
            embeddings = self.embedding_model.encode(chunks)
            
            # Prepare points for Qdrant
            points = []
            document_id = hashlib.md5(content.encode()).hexdigest()
            
            chunk_start = 0
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = f"{document_id}_{i}"
                chunk_end = chunk_start + len(chunk)
                
                # Map chunk to appropriate citation
                chunk_citation = self.map_chunk_to_citation(
                    chunk, i, document_citations, chunk_start, chunk_end
                )
                
                point_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "content": chunk,
                    "domain": domain,
                    "source": source or "unknown",
                    "source_type": source_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "citation": chunk_citation,
                    "chunk_start_pos": chunk_start,
                    "chunk_end_pos": chunk_end,
                    **metadata
                }
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=point_metadata
                    )
                )
                
                chunk_start = chunk_end
            
            # Insert into Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            # Update BM25 index with new chunks
            for chunk in chunks:
                tokens = self.preprocess_text(chunk)
                self.bm25_corpus.append(tokens)
                self.bm25_documents.append({
                    "id": f"{document_id}_{len(self.bm25_documents)}",
                    "content": chunk,
                    "payload": {"document_id": document_id, "domain": domain, "source": source or "unknown"}
                })
            
            # Rebuild BM25 index if we have documents
            if self.bm25_corpus:
                self.bm25_index = BM25Okapi(self.bm25_corpus)
            
            logger.info(f"Added document {document_id} with {len(chunks)} chunks")
            
            return {
                "document_id": document_id,
                "chunks_created": len(chunks),
                "embedding_dimension": self.embedding_dimension
            }
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    async def search(self, query: str, domain_weights: Dict[str, float] = None, k: int = 5, threshold: float = 0.7, 
                    use_bm25_reranking: bool = True, bm25_weight: float = 0.3, embedding_weight: float = 0.7) -> Dict[str, Any]:
        """Search for relevant documents with BM25 + Embedding reranking"""
        start_time = datetime.utcnow()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Get initial candidates from vector search (more than needed for reranking)
            search_limit = max(k * 4, 20) if use_bm25_reranking else k * 2
            
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=search_limit,
                score_threshold=threshold * 0.8  # Lower threshold for initial retrieval
            )
            
            # Prepare candidates for reranking
            candidates = []
            for result in search_results:
                chunk_content = result.payload.get("content", "")
                domain_score = self.calculate_domain_score(chunk_content, domain_weights or {})
                
                candidate = {
                    "content": chunk_content,
                    "embedding_score": float(result.score),
                    "domain_score": float(domain_score),
                    "metadata": result.payload,
                    "document_id": result.payload.get("document_id"),
                    "source": result.payload.get("source"),
                    "source_type": result.payload.get("source_type", "text"),
                    "citation": result.payload.get("citation"),
                    "bm25_score": 0.0,
                    "combined_score": 0.0
                }
                candidates.append(candidate)
            
            # Apply BM25 reranking if enabled and index is available
            if use_bm25_reranking and self.bm25_index and self.bm25_corpus:
                try:
                    # Preprocess query for BM25
                    query_tokens = self.preprocess_text(query)
                    
                    # Get BM25 scores for all documents
                    bm25_scores = self.bm25_index.get_scores(query_tokens)
                    
                    # Map BM25 scores to candidates
                    for i, candidate in enumerate(candidates):
                        # Find matching document in BM25 index
                        candidate_content = candidate["content"]
                        best_bm25_score = 0.0
                        
                        for j, bm25_doc in enumerate(self.bm25_documents):
                            if bm25_doc["content"] == candidate_content:
                                best_bm25_score = float(bm25_scores[j]) if j < len(bm25_scores) else 0.0
                                break
                        
                        candidate["bm25_score"] = best_bm25_score
                    
                    # Normalize BM25 scores
                    bm25_scores_list = [c["bm25_score"] for c in candidates]
                    if bm25_scores_list and max(bm25_scores_list) > 0:
                        max_bm25 = max(bm25_scores_list)
                        for candidate in candidates:
                            candidate["bm25_score"] = candidate["bm25_score"] / max_bm25
                    
                    logger.debug(f"Applied BM25 reranking to {len(candidates)} candidates")
                    
                except Exception as e:
                    logger.warning(f"BM25 reranking failed, falling back to embedding-only: {e}")
                    use_bm25_reranking = False
            
            # Calculate combined scores
            for candidate in candidates:
                if use_bm25_reranking:
                    # Combine embedding, BM25, and domain scores
                    combined_score = (
                        candidate["embedding_score"] * embedding_weight +
                        candidate["bm25_score"] * bm25_weight +
                        candidate["domain_score"] * 0.2
                    )
                else:
                    # Fallback to embedding + domain scoring
                    combined_score = candidate["embedding_score"] * 0.8 + candidate["domain_score"] * 0.2
                
                candidate["combined_score"] = combined_score
            
            # Sort by combined score and apply final threshold
            candidates.sort(key=lambda x: x["combined_score"], reverse=True)
            final_results = [c for c in candidates[:k] if c["combined_score"] >= threshold]
            
            # Generate evidence text
            evidence_chunks = [r["content"] for r in final_results]
            evidence = "\n\n".join(evidence_chunks)
            
            search_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"Search completed in {search_time:.3f}s. Found {len(final_results)} relevant chunks. "
                       f"BM25 reranking: {'enabled' if use_bm25_reranking else 'disabled'}")
            
            return {
                "query": query,
                "results": final_results,
                "evidence": evidence,
                "search_time": search_time,
                "reranking_method": "bm25_embedding" if use_bm25_reranking else "embedding_only"
            }
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

# Global service instance
rag_service = RAGService()

@api.on_event("startup")
async def startup_event():
    """Initialize RAG service on startup"""
    await rag_service.initialize()

@api.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Qdrant connection
        collections = rag_service.qdrant_client.get_collections() if rag_service.qdrant_client else None
        qdrant_status = "connected" if collections else "disconnected"
    except:
        qdrant_status = "error"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "embedding_model_loaded": rag_service.embedding_model is not None,
        "qdrant_status": qdrant_status,
        "embedding_dimension": rag_service.embedding_dimension
    }

@api.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for relevant documents"""
    try:
        result = await rag_service.search(
            query=request.query,
            domain_weights=request.domain_weights,
            k=request.k,
            threshold=request.threshold,
            use_bm25_reranking=request.use_bm25_reranking,
            bm25_weight=request.bm25_weight,
            embedding_weight=request.embedding_weight
        )
        
        return SearchResponse(**result)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@api.post("/documents", response_model=DocumentResponse)
async def add_document(request: DocumentRequest):
    """Add a document to the knowledge base"""
    try:
        result = await rag_service.add_document(
            content=request.content,
            metadata=request.metadata,
            domain=request.domain,
            source=request.source,
            source_type=request.source_type,
            timestamps=request.timestamps,
            citations=request.citations
        )
        
        return DocumentResponse(**result)
        
    except Exception as e:
        logger.error(f"Document addition error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")

@api.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    domain: str = "general",
    metadata: str = "{}"
):
    """Upload a document file"""
    try:
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata)
        except:
            metadata_dict = {}
        
        # Add file info to metadata
        metadata_dict.update({
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(content)
        })
        
        result = await rag_service.add_document(
            content=text_content,
            metadata=metadata_dict,
            domain=domain,
            source=file.filename,
            source_type="file"
        )
        
        return DocumentResponse(**result)
        
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@api.post("/documents/asr", response_model=DocumentResponse)
async def add_asr_document(
    transcript: str,
    segments: List[Dict[str, Any]],
    technical_segments: List[Dict[str, Any]],
    domain: str = "general",
    source: str = "asr_transcription",
    use_technical_only: bool = True
):
    """Add ASR transcription with timestamp citations to the knowledge base"""
    try:
        # Use either technical segments only or all segments based on preference
        segments_to_use = technical_segments if (use_technical_only and technical_segments) else segments
        
        # Prepare metadata
        metadata = {
            "total_segments": len(segments),
            "technical_segments": len(technical_segments),
            "use_technical_only": use_technical_only,
            "audio_duration": segments[-1].get("end", 0.0) if segments else 0.0
        }
        
        # Use segments as timestamps for citation generation
        timestamps = segments_to_use
        
        # Build content from selected segments
        if use_technical_only and technical_segments:
            content = " ".join([seg.get("text", "") for seg in technical_segments])
        else:
            content = transcript
        
        result = await rag_service.add_document(
            content=content,
            metadata=metadata,
            domain=domain,
            source=source,
            source_type="asr",
            timestamps=timestamps
        )
        
        return DocumentResponse(**result)
        
    except Exception as e:
        logger.error(f"ASR document addition error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add ASR document: {str(e)}")

@api.get("/collections/info")
async def get_collection_info():
    """Get information about the document collection"""
    try:
        if not rag_service.qdrant_client:
            raise HTTPException(status_code=503, detail="Qdrant client not initialized")
        
        collection_info = rag_service.qdrant_client.get_collection(rag_service.collection_name)
        
        return {
            "collection_name": rag_service.collection_name,
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count,
            "status": collection_info.status,
            "config": {
                "dimension": rag_service.embedding_dimension,
                "distance": "cosine"
            }
        }
        
    except Exception as e:
        logger.error(f"Collection info error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")

@api.delete("/collections/clear")
async def clear_collection():
    """Clear all documents from the collection (use with caution)"""
    try:
        if not rag_service.qdrant_client:
            raise HTTPException(status_code=503, detail="Qdrant client not initialized")
        
        # Delete and recreate collection
        rag_service.qdrant_client.delete_collection(rag_service.collection_name)
        await rag_service.ensure_collection_exists()
        
        return {"success": True, "message": "Collection cleared successfully"}
        
    except Exception as e:
        logger.error(f"Collection clear error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear collection: {str(e)}")

@api.get("/citations/{document_id}")
async def get_document_citations(document_id: str):
    """Get all citations for a specific document"""
    try:
        if not rag_service.qdrant_client:
            raise HTTPException(status_code=503, detail="Qdrant client not initialized")
        
        # Search for all chunks of this document
        search_result = rag_service.qdrant_client.scroll(
            collection_name=rag_service.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            ),
            limit=1000
        )
        
        citations = []
        for point in search_result[0]:
            citation = point.payload.get("citation")
            if citation:
                citation_info = {
                    "chunk_id": point.id,
                    "chunk_index": point.payload.get("chunk_index"),
                    "content": point.payload.get("content"),
                    "citation": citation
                }
                citations.append(citation_info)
        
        return {
            "document_id": document_id,
            "citations": citations,
            "total_chunks": len(citations)
        }
        
    except Exception as e:
        logger.error(f"Citations retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get citations: {str(e)}")

@api.get("/citations/search")
async def search_by_citation(
    source: Optional[str] = None,
    source_type: Optional[str] = None,
    technical_only: bool = False,
    limit: int = 100
):
    """Search documents by citation criteria"""
    try:
        if not rag_service.qdrant_client:
            raise HTTPException(status_code=503, detail="Qdrant client not initialized")
        
        # Build filter conditions
        filter_conditions = []
        
        if source:
            filter_conditions.append(
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=source)
                )
            )
        
        if source_type:
            filter_conditions.append(
                models.FieldCondition(
                    key="source_type",
                    match=models.MatchValue(value=source_type)
                )
            )
        
        if technical_only:
            filter_conditions.append(
                models.FieldCondition(
                    key="citation.is_technical",
                    match=models.MatchValue(value=True)
                )
            )
        
        # Perform search
        search_filter = models.Filter(must=filter_conditions) if filter_conditions else None
        
        search_result = rag_service.qdrant_client.scroll(
            collection_name=rag_service.collection_name,
            scroll_filter=search_filter,
            limit=limit
        )
        
        results = []
        for point in search_result[0]:
            result = {
                "chunk_id": point.id,
                "content": point.payload.get("content"),
                "source": point.payload.get("source"),
                "source_type": point.payload.get("source_type"),
                "citation": point.payload.get("citation"),
                "domain": point.payload.get("domain"),
                "timestamp": point.payload.get("timestamp")
            }
            results.append(result)
        
        return {
            "results": results,
            "total_found": len(results),
            "search_criteria": {
                "source": source,
                "source_type": source_type,
                "technical_only": technical_only
            }
        }
        
    except Exception as e:
        logger.error(f"Citation search error: {e}")
        raise HTTPException(status_code=500, detail=f"Citation search failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)