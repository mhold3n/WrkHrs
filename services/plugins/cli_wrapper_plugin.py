"""
CLI Tool Wrapper Plugin - External Command Line Tool Integration

This plugin demonstrates how to wrap external CLI tools and integrate them
into the AI stack with auto-discovery support. It provides a framework for
executing command-line tools safely and parsing their output.
"""

import subprocess
import json
import shlex
from typing import Dict, Any, List, Optional
from pluggy import HookspecMarker, HookimplMarker

# Plugin metadata
PLUGIN_INFO = {
    "name": "cli_wrapper",
    "version": "1.0.0",
    "description": "Wrapper for external command-line tools with safety and parsing",
    "author": "AI Stack Team",
    "category": "system",
    "domains": ["general", "chemistry", "mechanical", "materials"],
    "tags": ["cli", "wrapper", "external", "tools", "system"]
}

# Pluggy markers
hookspec = HookspecMarker("ai_stack")
hookimpl = HookimplMarker("ai_stack")

# Allowed CLI commands for security (whitelist approach)
ALLOWED_COMMANDS = {
    "echo": {
        "binary": "echo",
        "description": "Echo text to output",
        "max_args": 10,
        "timeout": 5
    },
    "date": {
        "binary": "date",
        "description": "Display current date and time",
        "max_args": 5,
        "timeout": 5
    },
    "whoami": {
        "binary": "whoami",
        "description": "Display current user",
        "max_args": 0,
        "timeout": 5
    },
    "pwd": {
        "binary": "pwd",
        "description": "Display current directory",
        "max_args": 0,
        "timeout": 5
    },
    "ls": {
        "binary": "ls",
        "description": "List directory contents",
        "max_args": 5,
        "timeout": 10
    },
    "wc": {
        "binary": "wc",
        "description": "Count lines, words, characters",
        "max_args": 5,
        "timeout": 10
    },
    "head": {
        "binary": "head",
        "description": "Display first lines of file/input",
        "max_args": 5,
        "timeout": 10
    },
    "tail": {
        "binary": "tail",
        "description": "Display last lines of file/input",
        "max_args": 5,
        "timeout": 10
    },
    "grep": {
        "binary": "grep",
        "description": "Search text patterns",
        "max_args": 10,
        "timeout": 15
    },
    "sort": {
        "binary": "sort",
        "description": "Sort lines of text",
        "max_args": 5,
        "timeout": 15
    },
    "uniq": {
        "binary": "uniq",
        "description": "Report or filter unique lines",
        "max_args": 5,
        "timeout": 10
    },
    "cut": {
        "binary": "cut",
        "description": "Extract sections from lines",
        "max_args": 5,
        "timeout": 10
    }
}


class CLIWrapperPlugin:
    """CLI wrapper plugin implementation"""
    
    @hookimpl
    def get_plugin_info(self) -> Dict[str, Any]:
        """Return plugin metadata"""
        return PLUGIN_INFO
    
    @hookimpl
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return list of available tools in this plugin"""
        tools = []
        
        for cmd_name, cmd_info in ALLOWED_COMMANDS.items():
            tools.append({
                "name": f"cli_{cmd_name}",
                "description": f"Execute {cmd_name}: {cmd_info['description']}",
                "parameters": {
                    "args": {
                        "type": "list", 
                        "description": f"Command arguments (max {cmd_info['max_args']})",
                        "optional": True
                    },
                    "input": {
                        "type": "string",
                        "description": "Input to pipe to command",
                        "optional": True
                    }
                },
                "returns": {
                    "type": "object",
                    "description": "Command execution result with stdout, stderr, and metadata"
                }
            })
        
        # Add general CLI executor
        tools.append({
            "name": "execute_cli",
            "description": "Execute allowed CLI commands with safety checks",
            "parameters": {
                "command": {"type": "string", "description": "Command to execute"},
                "args": {"type": "list", "description": "Command arguments", "optional": True},
                "input": {"type": "string", "description": "Input to pipe", "optional": True},
                "timeout": {"type": "int", "description": "Timeout in seconds", "optional": True}
            },
            "returns": {
                "type": "object",
                "description": "Execution result with output and metadata"
            }
        })
        
        return tools
    
    @hookimpl
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool with given parameters"""
        try:
            if tool_name == "execute_cli":
                return self.execute_cli(
                    command=parameters["command"],
                    args=parameters.get("args", []),
                    input_data=parameters.get("input"),
                    timeout=parameters.get("timeout", 30)
                )
            elif tool_name.startswith("cli_"):
                # Extract command name from tool name
                cmd_name = tool_name[4:]  # Remove "cli_" prefix
                if cmd_name in ALLOWED_COMMANDS:
                    return self.execute_cli(
                        command=cmd_name,
                        args=parameters.get("args", []),
                        input_data=parameters.get("input")
                    )
                else:
                    return {
                        "success": False,
                        "error": f"Unknown command: {cmd_name}",
                        "result": None
                    }
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "result": None
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
    
    def execute_cli(self, command: str, args: List[str] = None, 
                   input_data: str = None, timeout: int = 30) -> Dict[str, Any]:
        """Execute CLI command with safety checks"""
        if command not in ALLOWED_COMMANDS:
            raise ValueError(f"Command '{command}' not allowed. Allowed commands: {list(ALLOWED_COMMANDS.keys())}")
        
        cmd_config = ALLOWED_COMMANDS[command]
        args = args or []
        
        # Validate arguments
        if len(args) > cmd_config["max_args"]:
            raise ValueError(f"Too many arguments for {command}. Max allowed: {cmd_config['max_args']}")
        
        # Use configured timeout
        exec_timeout = min(timeout, cmd_config["timeout"])
        
        # Build command
        cmd_parts = [cmd_config["binary"]] + args
        
        # Sanitize arguments to prevent injection
        safe_args = []
        for arg in args:
            # Basic sanitization - in production, use more sophisticated validation
            if any(char in arg for char in [';', '&', '|', '`', '$', '(', ')', '<', '>', '\n']):
                raise ValueError(f"Potentially unsafe argument: {arg}")
            safe_args.append(shlex.quote(str(arg)))
        
        safe_cmd = [cmd_config["binary"]] + safe_args
        
        try:
            # Execute command
            process = subprocess.run(
                safe_cmd,
                input=input_data,
                text=True,
                capture_output=True,
                timeout=exec_timeout,
                check=False  # Don't raise exception on non-zero exit code
            )
            
            result = {
                "command": command,
                "args": args,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "exit_code": process.returncode,
                "execution_time": "N/A",  # Could add timing if needed
                "success": process.returncode == 0
            }
            
            return {
                "success": True,
                "error": None,
                "result": result
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command '{command}' timed out after {exec_timeout} seconds",
                "result": None
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": f"Command '{command}' not found on system",
                "result": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {str(e)}",
                "result": None
            }
    
    def get_allowed_commands(self) -> Dict[str, Dict[str, Any]]:
        """Get list of allowed commands with their configurations"""
        return ALLOWED_COMMANDS.copy()
    
    def is_command_safe(self, command: str, args: List[str] = None) -> bool:
        """Check if a command and arguments are safe to execute"""
        if command not in ALLOWED_COMMANDS:
            return False
        
        if args:
            cmd_config = ALLOWED_COMMANDS[command]
            if len(args) > cmd_config["max_args"]:
                return False
            
            # Check for unsafe characters
            for arg in args:
                if any(char in str(arg) for char in [';', '&', '|', '`', '$', '(', ')', '<', '>', '\n']):
                    return False
        
        return True
    
    def parse_command_output(self, output: str, format_type: str = "lines") -> Any:
        """Parse command output into structured format"""
        if format_type == "lines":
            return output.strip().split('\n') if output.strip() else []
        elif format_type == "words":
            return output.strip().split() if output.strip() else []
        elif format_type == "json":
            try:
                return json.loads(output) if output.strip() else {}
            except json.JSONDecodeError:
                return {"error": "Invalid JSON output", "raw": output}
        elif format_type == "csv":
            lines = output.strip().split('\n')
            return [line.split(',') for line in lines] if lines else []
        else:
            return output


# Plugin instance for auto-discovery
plugin_instance = CLIWrapperPlugin()


def get_plugin():
    """Entry point for plugin discovery"""
    return plugin_instance


# For direct usage
if __name__ == "__main__":
    wrapper = CLIWrapperPlugin()
    
    # Test the CLI wrapper
    print("CLI Wrapper Plugin Test")
    print("=" * 25)
    
    # Test basic commands
    print("Testing echo command:")
    result = wrapper.execute_cli("echo", ["Hello", "World"])
    print(f"Result: {result}")
    
    print("\nTesting date command:")
    result = wrapper.execute_cli("date")
    print(f"Result: {result}")
    
    print("\nTesting whoami command:")
    result = wrapper.execute_cli("whoami")
    print(f"Result: {result}")
    
    # Test with input
    print("\nTesting wc with input:")
    result = wrapper.execute_cli("wc", ["-w"], input_data="This is a test string with words")
    print(f"Result: {result}")
    
    # Test safety check
    print("\nTesting safety check:")
    safe = wrapper.is_command_safe("echo", ["safe", "argument"])
    unsafe = wrapper.is_command_safe("echo", ["unsafe; rm -rf /"])
    print(f"Safe command check: {safe}")
    print(f"Unsafe command check: {unsafe}")
    
    # Show allowed commands
    print(f"\nAllowed commands: {list(wrapper.get_allowed_commands().keys())}")