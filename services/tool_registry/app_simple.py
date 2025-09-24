import os
import json
import yaml
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import importlib.util
import inspect

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/logs/tool_registry.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
api = FastAPI(
    title="Tool Registry",
    description="Simple tool registry for AI stack",
    version="1.0.0"
)

class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = {}
    category: str = "general"
    version: str = "1.0.0"

class ToolExecution(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = {}

class ToolRegistry:
    """Simple tool registry without pluggy"""
    
    def __init__(self, plugins_dir: str = "/plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.tools = {}
        
        # Ensure plugins directory exists
        self.plugins_dir.mkdir(exist_ok=True)
        
        # Load built-in tools
        self._load_builtin_tools()
        
        # Load external plugins
        self._load_plugins()
    
    def _load_builtin_tools(self):
        """Load built-in tools"""
        builtin_tools = {
            "calculator": {
                "name": "calculator",
                "description": "Basic mathematical calculations",
                "parameters": {
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                },
                "category": "math",
                "version": "1.0.0"
            },
            "text_processor": {
                "name": "text_processor", 
                "description": "Basic text processing operations",
                "parameters": {
                    "text": {"type": "string", "description": "Text to process"},
                    "operation": {"type": "string", "description": "Operation to perform (upper, lower, length)"}
                },
                "category": "text",
                "version": "1.0.0"
            }
        }
        
        for tool_name, tool_info in builtin_tools.items():
            self.tools[tool_name] = ToolInfo(**tool_info)
            logger.info(f"Loaded built-in tool: {tool_name}")
    
    def _load_plugins(self):
        """Load external plugin tools"""
        try:
            for plugin_file in self.plugins_dir.glob("*.py"):
                if plugin_file.name.startswith("__"):
                    continue
                    
                try:
                    spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Look for tool classes
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and hasattr(obj, 'get_tool_info'):
                            try:
                                tool_instance = obj()
                                tool_info = tool_instance.get_tool_info()
                                self.tools[tool_info['name']] = ToolInfo(**tool_info)
                                logger.info(f"Loaded plugin tool: {tool_info['name']}")
                            except Exception as e:
                                logger.error(f"Error loading tool from {plugin_file}: {e}")
                                
                except Exception as e:
                    logger.error(f"Error loading plugin {plugin_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error scanning plugins directory: {e}")
    
    def get_tools(self) -> List[ToolInfo]:
        """Get all registered tools"""
        return list(self.tools.values())
    
    def get_tool(self, tool_name: str) -> Optional[ToolInfo]:
        """Get specific tool by name"""
        return self.tools.get(tool_name)
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given parameters"""
        if tool_name not in self.tools:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        
        # Built-in tool execution
        if tool_name == "calculator":
            try:
                # Simple expression evaluation (be careful in production!)
                result = eval(parameters.get("expression", ""))
                return {"result": result, "status": "success"}
            except Exception as e:
                return {"error": str(e), "status": "error"}
        
        elif tool_name == "text_processor":
            text = parameters.get("text", "")
            operation = parameters.get("operation", "length")
            
            if operation == "upper":
                result = text.upper()
            elif operation == "lower":
                result = text.lower()
            elif operation == "length":
                result = len(text)
            else:
                result = f"Unknown operation: {operation}"
            
            return {"result": result, "status": "success"}
        
        # Try to find and execute plugin tool
        try:
            for plugin_file in self.plugins_dir.glob("*.py"):
                if plugin_file.name.startswith("__"):
                    continue
                    
                spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and hasattr(obj, 'get_tool_info'):
                        try:
                            tool_instance = obj()
                            tool_info = tool_instance.get_tool_info()
                            if tool_info['name'] == tool_name and hasattr(tool_instance, 'execute_tool'):
                                return tool_instance.execute_tool(parameters)
                        except Exception as e:
                            logger.error(f"Error executing tool {tool_name}: {e}")
                            
        except Exception as e:
            logger.error(f"Error executing plugin tool {tool_name}: {e}")
        
        return {"error": f"Tool '{tool_name}' execution not implemented", "status": "error"}

# Initialize registry
registry = ToolRegistry()

@api.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@api.get("/tools", response_model=List[ToolInfo])
async def list_tools():
    """List all available tools"""
    return registry.get_tools()

@api.get("/tools/{tool_name}", response_model=ToolInfo)
async def get_tool(tool_name: str):
    """Get specific tool information"""
    tool = registry.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    return tool

@api.post("/tools/{tool_name}/execute")
async def execute_tool(tool_name: str, execution: ToolExecution):
    """Execute a tool with parameters"""
    try:
        result = registry.execute_tool(tool_name, execution.parameters)
        return result
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "Tool Registry",
        "version": "1.0.0",
        "tools_count": len(registry.tools),
        "endpoints": {
            "health": "/health",
            "list_tools": "/tools",
            "get_tool": "/tools/{tool_name}",
            "execute_tool": "/tools/{tool_name}/execute"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)
