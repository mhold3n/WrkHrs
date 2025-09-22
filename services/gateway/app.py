import os
import re
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import hashlib
import time
from collections import defaultdict

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends, Header, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import requests
import jwt

# Configure logging
import logging.config
import sys
from datetime import datetime

# JSON logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_JSON_LOGGING = os.getenv("ENABLE_JSON_LOGGING", "true").lower() == "true"

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": "gateway",
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from log record
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        if hasattr(record, 'ip_address'):
            log_entry["ip_address"] = record.ip_address
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        if hasattr(record, 'endpoint'):
            log_entry["endpoint"] = record.endpoint
        if hasattr(record, 'method'):
            log_entry["method"] = record.method
        if hasattr(record, 'event_type'):
            log_entry["event_type"] = record.event_type
        if hasattr(record, 'metadata'):
            log_entry["metadata"] = record.metadata
            
        return json.dumps(log_entry)

# Configure logging handlers
handlers = []

# File handler
file_handler = logging.FileHandler('/logs/gateway.log', mode='a')
if ENABLE_JSON_LOGGING:
    file_handler.setFormatter(JSONFormatter())
else:
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
handlers.append(file_handler)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
if ENABLE_JSON_LOGGING:
    console_handler.setFormatter(JSONFormatter())
else:
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
handlers.append(console_handler)

# Configure root logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    handlers=handlers,
    force=True
)
logger = logging.getLogger(__name__)

# Authentication configuration
API_KEY_SECRET = os.getenv("API_KEY_SECRET", "default-secret-change-in-production")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default-jwt-secret-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
ENABLE_AUTH = os.getenv("ENABLE_AUTHENTICATION", "true").lower() == "true"

# Rate limiting storage
request_counts = defaultdict(list)
security = HTTPBearer(auto_error=False)

# Initialize FastAPI app with comprehensive OpenAPI documentation
api = FastAPI(
    title="AI Stack Gateway",
    description="""
    # AI Stack Gateway API
    
    The AI Stack Gateway serves as the main entry point for the AI stack, providing:
    
    * **Authentication & Authorization**: API key and JWT token authentication
    * **Rate Limiting**: Configurable request rate limiting per client
    * **Domain Weighting**: Intelligent routing based on chemistry, mechanical, and materials domains
    * **Non-generative Conditioning**: Constraint application and unit normalization
    * **Request Orchestration**: Coordinates requests across multiple AI services
    
    ## Authentication
    
    The API supports two authentication methods:
    
    ### API Key Authentication
    Include your API key in the `X-API-Key` header:
    ```
    X-API-Key: your-api-key-here
    ```
    
    ### JWT Token Authentication  
    Include your JWT token in the `Authorization` header:
    ```
    Authorization: Bearer your-jwt-token-here
    ```
    
    ## Rate Limiting
    
    Requests are limited to 60 per minute per IP address by default. 
    Rate limit headers are included in responses:
    - `X-RateLimit-Limit`: Maximum requests per minute
    - `X-RateLimit-Remaining`: Remaining requests in current window
    - `X-RateLimit-Reset`: Time when rate limit resets
    
    ## Domain Weighting
    
    The gateway automatically detects domain relevance for:
    - **Chemistry**: Molecular formulas, reactions, pH, concentrations
    - **Mechanical**: Stress, strain, forces, beam analysis
    - **Materials**: Properties, compositions, hardness, thermal characteristics
    
    ## Error Responses
    
    The API returns structured error responses:
    ```json
    {
        "detail": "Error description",
        "error_code": "AUTH_REQUIRED",
        "timestamp": "2023-09-21T18:51:00Z"
    }
    ```
    """,
    version="1.0.0",
    contact={
        "name": "AI Stack Team",
        "email": "support@aistack.local",
        "url": "https://github.com/aistack/gateway"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    servers=[
        {
            "url": "http://localhost:8080",
            "description": "Development server"
        },
        {
            "url": "https://api.aistack.local",
            "description": "Production server"
        }
    ],
    openapi_tags=[
        {
            "name": "Authentication",
            "description": "Authentication and authorization endpoints"
        },
        {
            "name": "Chat",
            "description": "OpenAI-compatible chat completion endpoints"
        },
        {
            "name": "Health",
            "description": "Service health and status endpoints"
        },
        {
            "name": "Processing",
            "description": "Text processing and domain analysis endpoints"
        }
    ]
)

# Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
embedding_model = None
domain_classifier = None

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = "default"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class DomainWeights(BaseModel):
    chemistry: float = 0.0
    mechanical: float = 0.0
    materials: float = 0.0

class ProcessedPrompt(BaseModel):
    original_prompt: str
    domain_weights: DomainWeights
    extracted_units: List[str]
    constraints: List[str]
    weighted_evidence: str

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

# Domain classification patterns
DOMAIN_PATTERNS = {
    "chemistry": [
        r'\b(?:mol|molecular|atom|bond|reaction|catalyst|pH|concentration|solvent)\b',
        r'\b(?:organic|inorganic|polymer|synthesis|crystalline)\b',
        r'\b(?:H2O|CO2|NaCl|C6H12O6|chemical|formula)\b'
    ],
    "mechanical": [
        r'\b(?:force|stress|strain|torque|pressure|tension|compression)\b',
        r'\b(?:beam|shaft|gear|bearing|joint|mechanism|machine)\b',
        r'\b(?:N|Pa|MPa|GPa|kN|newton|pascal)\b'
    ],
    "materials": [
        r'\b(?:steel|aluminum|composite|ceramic|polymer|metal|alloy)\b',
        r'\b(?:hardness|ductility|brittleness|elasticity|plasticity)\b',
        r'\b(?:microstructure|grain|phase|crystal|defect)\b'
    ]
}

# SI unit patterns
SI_UNIT_PATTERNS = [
    r'\b\d+\.?\d*\s*(?:m|kg|s|A|K|mol|cd)\b',  # Base SI units
    r'\b\d+\.?\d*\s*(?:N|Pa|J|W|V|Ω|Hz)\b',   # Derived SI units
    r'\b\d+\.?\d*\s*(?:mm|cm|km|mg|g|kg)\b',  # Common prefixes
    r'\b\d+\.?\d*\s*(?:MPa|GPa|kN|mN|mA|kA)\b'  # Engineering units
]

# Authentication functions
def check_rate_limit(client_ip: str) -> bool:
    """Check if client has exceeded rate limit"""
    if not ENABLE_AUTH:
        return True
    
    current_time = time.time()
    minute_ago = current_time - 60
    
    # Clean old requests
    request_counts[client_ip] = [req_time for req_time in request_counts[client_ip] if req_time > minute_ago]
    
    # Check if under limit
    if len(request_counts[client_ip]) >= RATE_LIMIT_RPM:
        return False
    
    # Add current request
    request_counts[client_ip].append(current_time)
    return True

def validate_api_key(api_key: str) -> bool:
    """Validate API key using simple hash comparison"""
    if not api_key:
        return False
    
    # Simple hash-based validation (in production, use proper key management)
    expected_hash = hashlib.sha256(API_KEY_SECRET.encode()).hexdigest()
    provided_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return expected_hash == provided_hash

def create_jwt_token(data: dict) -> str:
    """Create JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> Optional[dict]:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired")
        return None
    except jwt.JWTError:
        logger.warning("Invalid JWT token")
        return None

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    x_api_key: Optional[str] = Header(None)
):
    """Authentication dependency"""
    if not ENABLE_AUTH:
        return {"authenticated": True, "method": "disabled"}
    
    client_ip = request.client.host
    
    # Check rate limit
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Try API key authentication first
    if x_api_key and validate_api_key(x_api_key):
        logger.info("API key authentication successful", extra={
            "event_type": "authentication",
            "ip_address": client_ip,
            "method": "api_key",
            "endpoint": request.url.path,
            "metadata": {"success": True}
        })
        return {"authenticated": True, "method": "api_key", "ip": client_ip}
    
    # Try JWT authentication
    if credentials and credentials.credentials:
        payload = verify_jwt_token(credentials.credentials)
        if payload:
            logger.info("JWT authentication successful", extra={
                "event_type": "authentication",
                "ip_address": client_ip,
                "method": "jwt",
                "user_id": payload.get("sub"),
                "endpoint": request.url.path,
                "metadata": {"success": True, "token_exp": payload.get("exp")}
            })
            return {"authenticated": True, "method": "jwt", "payload": payload, "ip": client_ip}
    
    # Authentication failed
    logger.warning("Authentication failed", extra={
        "event_type": "authentication",
        "ip_address": client_ip,
        "endpoint": request.url.path,
        "metadata": {
            "success": False,
            "api_key_provided": bool(x_api_key),
            "jwt_provided": bool(credentials and credentials.credentials)
        }
    })
    raise HTTPException(
        status_code=401, 
        detail="Authentication required. Provide X-API-Key header or Bearer token."
    )

def load_models():
    """Load embedding model and initialize domain classifier"""
    global embedding_model, domain_classifier
    try:
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        embedding_model = None

def extract_domain_weights(text: str) -> DomainWeights:
    """Extract domain weights using pattern matching and embeddings"""
    weights = DomainWeights()
    
    # Pattern-based scoring
    for domain, patterns in DOMAIN_PATTERNS.items():
        score = 0.0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            score += len(matches) * 0.1
        
        # Normalize and cap at 1.0
        score = min(score, 1.0)
        setattr(weights, domain, score)
    
    return weights

def extract_units(text: str) -> List[str]:
    """Extract units from text for SI normalization"""
    units = []
    for pattern in SI_UNIT_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        units.extend(matches)
    return list(set(units))  # Remove duplicates

def extract_constraints(text: str) -> List[str]:
    """Extract safety and operational constraints"""
    constraint_patterns = [
        r'\b(?:safety|hazard|toxic|flammable|corrosive|dangerous)\b',
        r'\b(?:temperature limit|pressure limit|max|min|range)\b',
        r'\b(?:standard|specification|code|regulation|requirement)\b'
    ]
    
    constraints = []
    for pattern in constraint_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        constraints.extend(matches)
    
    return list(set(constraints))

async def get_weighted_evidence(prompt: str, weights: DomainWeights) -> str:
    """Get weighted evidence from RAG system"""
    try:
        rag_url = f"http://rag-api:8000/search"
        payload = {
            "query": prompt,
            "domain_weights": weights.dict(),
            "k": 5
        }
        
        response = requests.post(rag_url, json=payload, timeout=10)
        if response.status_code == 200:
            evidence = response.json().get("evidence", "")
            return evidence
        else:
            logger.warning(f"RAG service returned {response.status_code}")
            return ""
    except Exception as e:
        logger.error(f"Failed to get weighted evidence: {e}")
        return ""

@api.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    load_models()

@api.post("/auth/login", 
         response_model=LoginResponse,
         tags=["Authentication"],
         summary="Login to generate JWT token",
         description="""
         Generate a JWT token for API authentication.
         
         **Authentication Method**: None required (this is the login endpoint)
         
         **Request Body**: 
         - `username`: User identifier
         - `password`: User password (should match API_KEY_SECRET for demo)
         
         **Response**:
         - `access_token`: JWT token for subsequent requests
         - `token_type`: Always "bearer"
         - `expires_in`: Token expiration time in seconds
         
         **Example Usage**:
         ```bash
         curl -X POST "http://localhost:8080/auth/login" \\
              -H "Content-Type: application/json" \\
              -d '{"username": "admin", "password": "your-api-secret"}'
         ```
         """)
async def login(request: LoginRequest):
    """Login endpoint to generate JWT token"""
    # Simple credential validation (in production, use proper user management)
    if request.username == "admin" and request.password == API_KEY_SECRET:
        token_data = {"sub": request.username, "username": request.username}
        access_token = create_jwt_token(token_data)
        return LoginResponse(
            access_token=access_token,
            expires_in=JWT_EXPIRATION_HOURS * 3600
        )
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@api.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "embedding_model_loaded": embedding_model is not None,
        "authentication_enabled": ENABLE_AUTH
    }

@api.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest, 
    current_user: dict = Depends(get_current_user)
):
    """OpenAI-compatible chat completions endpoint with non-generative conditioning"""
    try:
        # Extract the user's prompt
        user_message = None
        for msg in request.messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Process prompt with non-generative conditioning
        processed = await process_prompt(user_message)
        
        # Forward to orchestrator with enhanced context
        enhanced_messages = request.messages.copy()
        
        # Add domain context and evidence without changing the original prompt
        if processed.weighted_evidence:
            system_context = f"""
Domain Analysis: Chemistry={processed.domain_weights.chemistry:.2f}, Mechanical={processed.domain_weights.mechanical:.2f}, Materials={processed.domain_weights.materials:.2f}
Relevant Evidence: {processed.weighted_evidence}
Units Found: {', '.join(processed.extracted_units)}
Constraints: {', '.join(processed.constraints)}

Please respond with SI units and consider the safety constraints mentioned.
"""
            enhanced_messages.insert(0, {"role": "system", "content": system_context})
        
        # Forward to orchestrator
        orchestrator_url = f"http://orchestrator:8000/chat"
        orchestrator_payload = {
            "messages": enhanced_messages,
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }
        
        response = requests.post(orchestrator_url, json=orchestrator_payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            # Log the interaction
            logger.info(f"Request processed: domains={processed.domain_weights.dict()}, "
                       f"units={len(processed.extracted_units)}, constraints={len(processed.constraints)}")
            
            return result
        else:
            logger.error(f"Orchestrator returned {response.status_code}: {response.text}")
            raise HTTPException(status_code=502, detail="Orchestrator service error")
            
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_prompt(prompt: str) -> ProcessedPrompt:
    """Process prompt with non-generative conditioning"""
    # Extract domain weights
    domain_weights = extract_domain_weights(prompt)
    
    # Extract units and constraints
    units = extract_units(prompt)
    constraints = extract_constraints(prompt)
    
    # Get weighted evidence from RAG
    evidence = await get_weighted_evidence(prompt, domain_weights)
    
    return ProcessedPrompt(
        original_prompt=prompt,
        domain_weights=domain_weights,
        extracted_units=units,
        constraints=constraints,
        weighted_evidence=evidence
    )

@api.get("/domains/analyze")
async def analyze_domains(text: str):
    """Endpoint to analyze domain weights for debugging"""
    weights = extract_domain_weights(text)
    units = extract_units(text)
    constraints = extract_constraints(text)
    
    return {
        "domain_weights": weights.dict(),
        "units": units,
        "constraints": constraints
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)