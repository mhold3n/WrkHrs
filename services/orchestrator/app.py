import os
import json
import logging
from typing import Dict, List, Optional, Any, Annotated
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/logs/orchestrator.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
api = FastAPI(
    title="AI Stack Orchestrator",
    description="LangGraph-based orchestration service",
    version="1.0.0"
)

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

class WorkflowState(BaseModel):
    messages: List[Dict[str, str]]
    current_step: str = "analyze"
    tools_needed: List[str] = []
    rag_results: Optional[str] = None
    asr_results: Optional[str] = None
    tool_results: Dict[str, Any] = {}
    final_response: Optional[str] = None

# Tool definitions
@tool
def search_knowledge_base(query: str, domain_weights: Dict[str, float] = None) -> str:
    """Search the RAG knowledge base with optional domain weighting."""
    try:
        rag_url = "http://rag-api:8000/search"
        payload = {
            "query": query,
            "domain_weights": domain_weights or {},
            "k": 5
        }
        
        response = requests.post(rag_url, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json().get("evidence", "No evidence found")
        else:
            return f"Error searching knowledge base: {response.status_code}"
    except Exception as e:
        logger.error(f"Error in search_knowledge_base: {e}")
        return f"Error searching knowledge base: {str(e)}"

@tool
def transcribe_audio(audio_data: str) -> str:
    """Transcribe audio/video content using ASR service."""
    try:
        asr_url = "http://asr-api:8000/transcribe"
        payload = {"audio_data": audio_data}
        
        response = requests.post(asr_url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get("transcript", "No transcript generated")
        else:
            return f"Error transcribing audio: {response.status_code}"
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {e}")
        return f"Error transcribing audio: {str(e)}"

@tool  
def get_domain_data(domain: str, query: str) -> str:
    """Get domain-specific data from MCP servers."""
    try:
        mcp_url = f"http://mcp:8000/{domain}/query"
        payload = {"query": query}
        
        response = requests.post(mcp_url, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json().get("data", "No domain data found")
        else:
            return f"Error getting domain data: {response.status_code}"
    except Exception as e:
        logger.error(f"Error in get_domain_data: {e}")
        return f"Error getting domain data: {str(e)}"

@tool
def get_available_tools() -> List[Dict[str, Any]]:
    """Get list of available tools from tool registry."""
    try:
        tools_url = "http://tool-registry:8000/tools"
        
        response = requests.get(tools_url, timeout=10)
        if response.status_code == 200:
            return response.json().get("tools", [])
        else:
            return []
    except Exception as e:
        logger.error(f"Error in get_available_tools: {e}")
        return []

def analyze_request(state: WorkflowState) -> WorkflowState:
    """Analyze the request to determine what tools are needed."""
    messages = state.messages
    user_message = ""
    
    # Find the user's message
    for msg in messages:
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    tools_needed = []
    
    # Determine if we need RAG search
    if any(keyword in user_message.lower() for keyword in 
           ["what is", "explain", "describe", "how", "why", "definition"]):
        tools_needed.append("rag_search")
    
    # Check for audio/video processing needs
    if any(keyword in user_message.lower() for keyword in 
           ["audio", "video", "transcript", "speech", "recording"]):
        tools_needed.append("asr")
    
    # Check for domain-specific queries
    domains = ["chemistry", "mechanical", "materials"]
    for domain in domains:
        if domain in user_message.lower():
            tools_needed.append(f"mcp_{domain}")
    
    state.tools_needed = tools_needed
    state.current_step = "gather_context" if tools_needed else "generate_response"
    
    logger.info(f"Analysis complete. Tools needed: {tools_needed}")
    return state

def gather_context(state: WorkflowState) -> WorkflowState:
    """Gather context from various sources based on analysis."""
    messages = state.messages
    user_message = ""
    
    # Find the user's message and extract domain weights from system message
    domain_weights = {}
    for msg in messages:
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
        elif msg.get("role") == "system":
            system_content = msg.get("content", "")
            # Parse domain weights from system message
            if "Domain Analysis:" in system_content:
                try:
                    lines = system_content.split('\n')
                    for line in lines:
                        if "Chemistry=" in line:
                            parts = line.split(',')
                            for part in parts:
                                if "Chemistry=" in part:
                                    domain_weights["chemistry"] = float(part.split('=')[1])
                                elif "Mechanical=" in part:
                                    domain_weights["mechanical"] = float(part.split('=')[1])
                                elif "Materials=" in part:
                                    domain_weights["materials"] = float(part.split('=')[1])
                except:
                    pass
    
    # Execute tools based on needs
    for tool_name in state.tools_needed:
        try:
            if tool_name == "rag_search":
                result = search_knowledge_base(user_message, domain_weights)
                state.rag_results = result
                state.tool_results["rag"] = result
                
            elif tool_name == "asr":
                # For demo purposes, assume audio data is passed somehow
                result = "Audio processing not implemented in demo"
                state.asr_results = result
                state.tool_results["asr"] = result
                
            elif tool_name.startswith("mcp_"):
                domain = tool_name.split("_")[1]
                result = get_domain_data(domain, user_message)
                state.tool_results[f"mcp_{domain}"] = result
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            state.tool_results[tool_name] = f"Error: {str(e)}"
    
    state.current_step = "generate_response"
    return state

def generate_response(state: WorkflowState) -> WorkflowState:
    """Generate final response using LLM."""
    try:
        # Prepare enhanced context
        context_parts = []
        
        if state.rag_results:
            context_parts.append(f"Knowledge Base Results:\n{state.rag_results}")
        
        if state.asr_results:
            context_parts.append(f"Audio Transcript:\n{state.asr_results}")
        
        for tool_name, result in state.tool_results.items():
            if tool_name.startswith("mcp_"):
                domain = tool_name.split("_")[1]
                context_parts.append(f"{domain.title()} Domain Data:\n{result}")
        
        # Build the final prompt
        enhanced_messages = state.messages.copy()
        if context_parts:
            context = "\n\n".join(context_parts)
            # Add context to system message or create one
            system_found = False
            for i, msg in enumerate(enhanced_messages):
                if msg.get("role") == "system":
                    enhanced_messages[i]["content"] += f"\n\nAdditional Context:\n{context}"
                    system_found = True
                    break
            
            if not system_found:
                enhanced_messages.insert(0, {
                    "role": "system", 
                    "content": f"Additional Context:\n{context}"
                })
        
        # Call LLM service (Ollama/vLLM)
        llm_backend = os.getenv("LLM_BACKEND", "ollama")
        
        if llm_backend == "ollama":
            llm_url = "http://llm-runner:11434/api/chat"
            payload = {
                "model": os.getenv("OLLAMA_MODEL", "llama3:8b-instruct"),
                "messages": enhanced_messages,
                "stream": False
            }
        else:  # vLLM
            llm_url = "http://llm-runner:8000/v1/chat/completions"
            payload = {
                "model": os.getenv("VLLM_MODEL", "default"),
                "messages": enhanced_messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
        
        response = requests.post(llm_url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            if llm_backend == "ollama":
                state.final_response = result.get("message", {}).get("content", "No response generated")
            else:  # vLLM
                choices = result.get("choices", [])
                if choices:
                    state.final_response = choices[0].get("message", {}).get("content", "No response generated")
                else:
                    state.final_response = "No response generated"
        else:
            logger.error(f"LLM service returned {response.status_code}: {response.text}")
            state.final_response = f"Error generating response: {response.status_code}"
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        state.final_response = f"Error generating response: {str(e)}"
    
    state.current_step = "complete"
    return state

# Create the workflow graph
def create_workflow():
    """Create the LangGraph workflow."""
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_request)
    workflow.add_node("gather_context", gather_context)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges
    workflow.set_entry_point("analyze")
    
    # Conditional routing from analyze
    def should_gather_context(state):
        return "gather_context" if state.tools_needed else "generate_response"
    
    workflow.add_conditional_edges("analyze", should_gather_context)
    workflow.add_edge("gather_context", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()

# Global workflow instance
workflow_app = create_workflow()

@api.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "workflow_ready": workflow_app is not None
    }

@api.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint that processes requests through the workflow."""
    try:
        # Initialize workflow state
        initial_state = WorkflowState(messages=request.messages)
        
        # Run the workflow
        result = workflow_app.invoke(initial_state)
        
        # Format response in OpenAI format
        response = ChatResponse(
            id=f"chatcmpl-{datetime.utcnow().timestamp()}",
            created=int(datetime.utcnow().timestamp()),
            model=request.model or "orchestrator",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.final_response or "No response generated"
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 0,  # Would need tokenizer to calculate
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )
        
        logger.info(f"Chat request processed successfully. Tools used: {result.tools_needed}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/workflow/status")
async def workflow_status():
    """Get workflow status and available tools."""
    try:
        tools = get_available_tools()
        return {
            "status": "active",
            "available_tools": len(tools),
            "workflow_nodes": ["analyze", "gather_context", "generate_response"]
        }
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)