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
import pluggy

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
    description="Pluggy-based tool auto-discovery and registry",
    version="1.0.0"
)

# Tool plugin specification
class ToolSpec:
    """Plugin specification for tools"""
    
    @pluggy.hookspec
    def get_tool_info(self) -> Dict[str, Any]:
        """Return tool information"""
        pass
    
    @pluggy.hookspec
    def execute_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass

class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = {}
    returns: Dict[str, Any] = {}
    category: str = "general"
    version: str = "1.0.0"
    author: str = "unknown"

class ToolExecutionRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = {}

class ToolExecutionResponse(BaseModel):
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0

class ToolRegistry:
    """Main tool registry managing plugins"""
    
    def __init__(self, plugins_dir: str = "/plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.pm = pluggy.PluginManager("tool_registry")
        self.pm.add_hookspecs(ToolSpec)
        self.tools = {}
        self.plugins = {}
        
        # Ensure plugins directory exists
        self.plugins_dir.mkdir(exist_ok=True)
        
        # Load plugins on initialization
        self.discover_and_load_plugins()
    
    def discover_and_load_plugins(self):
        """Discover and load all plugins from the plugins directory"""
        logger.info(f"Scanning for plugins in {self.plugins_dir}")
        
        # Clear existing tools
        self.tools = {}
        self.plugins = {}
        
        # Load Python plugins
        self._load_python_plugins()
        
        # Load YAML manifest plugins
        self._load_yaml_plugins()
        
        # Load CLI wrapper plugins
        self._load_cli_plugins()
        
        logger.info(f"Loaded {len(self.tools)} tools from {len(self.plugins)} plugins")
    
    def _load_python_plugins(self):
        """Load Python-based plugins"""
        for py_file in self.plugins_dir.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue
                
            try:
                # Load module dynamically
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find plugin classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        hasattr(obj, 'get_tool_info') and 
                        hasattr(obj, 'execute_tool')):
                        
                        plugin_instance = obj()
                        self.pm.register(plugin_instance, name=name)
                        
                        # Get tool info
                        tool_info = plugin_instance.get_tool_info()
                        tool_name = tool_info.get('name', name)
                        
                        self.tools[tool_name] = {
                            'info': tool_info,
                            'plugin': plugin_instance,
                            'type': 'python'
                        }
                        self.plugins[name] = plugin_instance
                        
                        logger.info(f"Loaded Python plugin: {tool_name}")
                        
            except Exception as e:
                logger.error(f"Failed to load Python plugin {py_file}: {e}")
    
    def _load_yaml_plugins(self):
        """Load YAML manifest-based plugins"""
        for yaml_file in self.plugins_dir.glob("**/*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    manifest = yaml.safe_load(f)
                
                if not isinstance(manifest, dict) or 'tool' not in manifest:
                    continue
                
                tool_config = manifest['tool']
                tool_name = tool_config.get('name')
                
                if not tool_name:
                    logger.warning(f"YAML plugin {yaml_file} missing tool name")
                    continue
                
                # Create wrapper for YAML-defined tools
                yaml_plugin = YAMLToolPlugin(yaml_file, manifest)
                
                self.tools[tool_name] = {
                    'info': tool_config,
                    'plugin': yaml_plugin,
                    'type': 'yaml'
                }
                self.plugins[tool_name] = yaml_plugin
                
                logger.info(f"Loaded YAML plugin: {tool_name}")
                
            except Exception as e:
                logger.error(f"Failed to load YAML plugin {yaml_file}: {e}")
    
    def _load_cli_plugins(self):
        """Load CLI wrapper plugins"""
        for json_file in self.plugins_dir.glob("**/*_cli.json"):
            try:
                with open(json_file, 'r') as f:
                    cli_config = json.load(f)
                
                tool_name = cli_config.get('name')
                if not tool_name:
                    logger.warning(f"CLI plugin {json_file} missing tool name")
                    continue
                
                # Create wrapper for CLI tools
                cli_plugin = CLIToolPlugin(json_file, cli_config)
                
                self.tools[tool_name] = {
                    'info': cli_config,
                    'plugin': cli_plugin,
                    'type': 'cli'
                }
                self.plugins[tool_name] = cli_plugin
                
                logger.info(f"Loaded CLI plugin: {tool_name}")
                
            except Exception as e:
                logger.error(f"Failed to load CLI plugin {json_file}: {e}")
    
    def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool by name"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        return [
            {
                'name': name,
                'type': tool_data['type'],
                **tool_data['info']
            }
            for name, tool_data in self.tools.items()
        ]
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given parameters"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        try:
            start_time = datetime.utcnow()
            result = tool['plugin'].execute_tool(parameters)
            end_time = datetime.utcnow()
            
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': 0.0
            }
    
    def refresh_plugins(self):
        """Refresh plugin discovery"""
        logger.info("Refreshing plugin registry")
        self.discover_and_load_plugins()

class YAMLToolPlugin:
    """Wrapper for YAML-defined tools"""
    
    def __init__(self, yaml_file: Path, manifest: Dict[str, Any]):
        self.yaml_file = yaml_file
        self.manifest = manifest
        self.tool_config = manifest['tool']
    
    def get_tool_info(self) -> Dict[str, Any]:
        return self.tool_config
    
    def execute_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        # For YAML tools, we might execute a command or script
        command = self.tool_config.get('command')
        if not command:
            raise ValueError("YAML tool missing command")
        
        # Basic command execution (could be enhanced with parameter substitution)
        import subprocess
        try:
            result = subprocess.run(
                command.split(), 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            raise RuntimeError("Tool execution timed out")

class CLIToolPlugin:
    """Wrapper for CLI tools"""
    
    def __init__(self, json_file: Path, cli_config: Dict[str, Any]):
        self.json_file = json_file
        self.cli_config = cli_config
    
    def get_tool_info(self) -> Dict[str, Any]:
        return self.cli_config
    
    def execute_tool(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        command_template = self.cli_config.get('command')
        if not command_template:
            raise ValueError("CLI tool missing command template")
        
        # Parameter substitution
        try:
            command = command_template.format(**parameters)
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")
        
        # Execute CLI command
        import subprocess
        try:
            result = subprocess.run(
                command, 
                shell=True,
                capture_output=True, 
                text=True, 
                timeout=60
            )
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'command': command
            }
        except subprocess.TimeoutExpired:
            raise RuntimeError("CLI tool execution timed out")

# Global registry instance
tool_registry = ToolRegistry()

@api.on_event("startup")
async def startup_event():
    """Initialize tool registry on startup"""
    tool_registry.discover_and_load_plugins()

@api.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "tools_loaded": len(tool_registry.tools),
        "plugins_loaded": len(tool_registry.plugins)
    }

@api.get("/tools")
async def list_tools():
    """List all available tools"""
    return {
        "tools": tool_registry.list_tools(),
        "count": len(tool_registry.tools)
    }

@api.get("/tools/{tool_name}")
async def get_tool_info(tool_name: str):
    """Get information about a specific tool"""
    tool = tool_registry.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    return {
        "name": tool_name,
        "type": tool['type'],
        **tool['info']
    }

@api.post("/tools/{tool_name}/execute", response_model=ToolExecutionResponse)
async def execute_tool(tool_name: str, request: ToolExecutionRequest):
    """Execute a specific tool"""
    try:
        result = tool_registry.execute_tool(tool_name, request.parameters)
        return ToolExecutionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

@api.post("/plugins/refresh")
async def refresh_plugins():
    """Refresh plugin discovery"""
    try:
        tool_registry.refresh_plugins()
        return {
            "success": True,
            "message": f"Refreshed plugins. Loaded {len(tool_registry.tools)} tools."
        }
    except Exception as e:
        logger.error(f"Plugin refresh error: {e}")
        raise HTTPException(status_code=500, detail=f"Plugin refresh failed: {str(e)}")

@api.get("/plugins/directory")
async def get_plugins_directory():
    """Get plugins directory information"""
    plugins_path = Path("/plugins")
    
    if not plugins_path.exists():
        return {"directory": str(plugins_path), "exists": False, "files": []}
    
    files = []
    for file_path in plugins_path.rglob("*"):
        if file_path.is_file():
            files.append({
                "name": file_path.name,
                "path": str(file_path.relative_to(plugins_path)),
                "size": file_path.stat().st_size,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
    
    return {
        "directory": str(plugins_path),
        "exists": True,
        "file_count": len(files),
        "files": files
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)