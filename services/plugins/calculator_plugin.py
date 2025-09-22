"""
Calculator Plugin - Basic mathematical operations

This plugin provides basic calculator functionality with auto-discovery support.
It demonstrates the plugin system architecture using decorators and metadata.
"""

import math
from typing import Dict, Any, List
from pluggy import HookspecMarker, HookimplMarker

# Plugin metadata
PLUGIN_INFO = {
    "name": "calculator",
    "version": "1.0.0",
    "description": "Basic mathematical operations calculator",
    "author": "AI Stack Team",
    "category": "mathematics",
    "domains": ["general", "engineering", "chemistry"],
    "tags": ["calculator", "math", "operations", "arithmetic"]
}

# Pluggy markers
hookspec = HookspecMarker("ai_stack")
hookimpl = HookimplMarker("ai_stack")


class CalculatorPlugin:
    """Calculator plugin implementation"""
    
    @hookimpl
    def get_plugin_info(self) -> Dict[str, Any]:
        """Return plugin metadata"""
        return PLUGIN_INFO
    
    @hookimpl
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return list of available tools in this plugin"""
        return [
            {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "a": {"type": "float", "description": "First number"},
                    "b": {"type": "float", "description": "Second number"}
                },
                "returns": {"type": "float", "description": "Sum of a and b"}
            },
            {
                "name": "subtract",
                "description": "Subtract second number from first",
                "parameters": {
                    "a": {"type": "float", "description": "First number"},
                    "b": {"type": "float", "description": "Second number"}
                },
                "returns": {"type": "float", "description": "Difference a - b"}
            },
            {
                "name": "multiply",
                "description": "Multiply two numbers",
                "parameters": {
                    "a": {"type": "float", "description": "First number"},
                    "b": {"type": "float", "description": "Second number"}
                },
                "returns": {"type": "float", "description": "Product of a and b"}
            },
            {
                "name": "divide",
                "description": "Divide first number by second",
                "parameters": {
                    "a": {"type": "float", "description": "Dividend"},
                    "b": {"type": "float", "description": "Divisor (cannot be zero)"}
                },
                "returns": {"type": "float", "description": "Quotient a / b"}
            },
            {
                "name": "power",
                "description": "Raise first number to the power of second",
                "parameters": {
                    "base": {"type": "float", "description": "Base number"},
                    "exponent": {"type": "float", "description": "Exponent"}
                },
                "returns": {"type": "float", "description": "Result of base^exponent"}
            },
            {
                "name": "sqrt",
                "description": "Calculate square root",
                "parameters": {
                    "x": {"type": "float", "description": "Number to calculate square root of"}
                },
                "returns": {"type": "float", "description": "Square root of x"}
            },
            {
                "name": "log",
                "description": "Calculate natural logarithm",
                "parameters": {
                    "x": {"type": "float", "description": "Number to calculate log of (must be > 0)"}
                },
                "returns": {"type": "float", "description": "Natural logarithm of x"}
            }
        ]
    
    @hookimpl
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool with given parameters"""
        try:
            if tool_name == "add":
                result = self.add(parameters["a"], parameters["b"])
            elif tool_name == "subtract":
                result = self.subtract(parameters["a"], parameters["b"])
            elif tool_name == "multiply":
                result = self.multiply(parameters["a"], parameters["b"])
            elif tool_name == "divide":
                result = self.divide(parameters["a"], parameters["b"])
            elif tool_name == "power":
                result = self.power(parameters["base"], parameters["exponent"])
            elif tool_name == "sqrt":
                result = self.sqrt(parameters["x"])
            elif tool_name == "log":
                result = self.log(parameters["x"])
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "result": None
                }
            
            return {
                "success": True,
                "error": None,
                "result": result
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        return float(a) + float(b)
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a"""
        return float(a) - float(b)
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        return float(a) * float(b)
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b"""
        b_float = float(b)
        if b_float == 0:
            raise ValueError("Division by zero is not allowed")
        return float(a) / b_float
    
    def power(self, base: float, exponent: float) -> float:
        """Calculate base^exponent"""
        return math.pow(float(base), float(exponent))
    
    def sqrt(self, x: float) -> float:
        """Calculate square root"""
        x_float = float(x)
        if x_float < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return math.sqrt(x_float)
    
    def log(self, x: float) -> float:
        """Calculate natural logarithm"""
        x_float = float(x)
        if x_float <= 0:
            raise ValueError("Logarithm input must be positive")
        return math.log(x_float)


# Plugin instance for auto-discovery
plugin_instance = CalculatorPlugin()


def get_plugin():
    """Entry point for plugin discovery"""
    return plugin_instance


# For direct usage
if __name__ == "__main__":
    calc = CalculatorPlugin()
    
    # Test the calculator
    print("Calculator Plugin Test")
    print("=" * 20)
    
    # Test basic operations
    print(f"2 + 3 = {calc.add(2, 3)}")
    print(f"10 - 4 = {calc.subtract(10, 4)}")
    print(f"5 * 6 = {calc.multiply(5, 6)}")
    print(f"15 / 3 = {calc.divide(15, 3)}")
    print(f"2^8 = {calc.power(2, 8)}")
    print(f"√16 = {calc.sqrt(16)}")
    print(f"ln(e) = {calc.log(math.e)}")
    
    # Test using execute_tool interface
    print("\nTesting execute_tool interface:")
    result = calc.execute_tool("add", {"a": 10, "b": 20})
    print(f"execute_tool('add', {{'a': 10, 'b': 20}}) = {result}")