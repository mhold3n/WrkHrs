"""
Web API Integration Plugin
External API integration for data enrichment and validation
"""

import json
import time
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlencode, quote
from dataclasses import dataclass
import hashlib

# Plugin metadata
PLUGIN_INFO = {
    "name": "web_api_integration",
    "version": "2.0.0",
    "description": "Integration with external APIs for data enrichment, validation, and scientific calculations",
    "author": "AI Stack Integration Team",
    "category": "integration",
    "domains": ["general", "chemistry", "materials", "data"],
    "tags": ["api", "web", "integration", "validation", "enrichment", "external"],
    "requires": ["json", "urllib", "hashlib", "time"],
    "security_note": "This plugin makes external HTTP requests. Ensure proper network security."
}

@dataclass
class APIResponse:
    """Standardized API response structure"""
    success: bool
    status_code: int
    data: Any
    error: Optional[str]
    response_time: float
    cached: bool = False

@dataclass
class APIEndpoint:
    """API endpoint configuration"""
    name: str
    base_url: str
    description: str
    requires_auth: bool
    rate_limit: int  # requests per minute
    timeout: int = 30

class APICache:
    """Simple in-memory cache for API responses"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
    
    def _generate_key(self, url: str, params: Dict) -> str:
        """Generate cache key from URL and parameters"""
        key_string = f"{url}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, url: str, params: Dict) -> Optional[Any]:
        """Get cached response if valid"""
        key = self._generate_key(url, params)
        
        if key in self.cache:
            timestamp = self.timestamps.get(key, 0)
            if time.time() - timestamp < self.ttl:
                return self.cache[key]
            else:
                # Remove expired entry
                del self.cache[key]
                del self.timestamps[key]
        
        return None
    
    def set(self, url: str, params: Dict, response: Any):
        """Cache API response"""
        key = self._generate_key(url, params)
        
        # Simple LRU: remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = response
        self.timestamps[key] = time.time()

class WebAPIPlugin:
    """Web API integration plugin for external data sources"""
    
    def __init__(self):
        self.cache = APICache()
        self.request_counts = {}  # Simple rate limiting
        
        # Define available API endpoints (demo/mock endpoints)
        self.endpoints = {
            "weather": APIEndpoint(
                name="Weather API",
                base_url="https://api.openweathermap.org/data/2.5",
                description="Weather data for locations",
                requires_auth=True,
                rate_limit=60
            ),
            "geocoding": APIEndpoint(
                name="Geocoding API", 
                base_url="https://nominatim.openstreetmap.org",
                description="Location coordinates from addresses",
                requires_auth=False,
                rate_limit=60
            ),
            "periodic_table": APIEndpoint(
                name="Periodic Table API",
                base_url="https://periodictable.p.rapidapi.com",
                description="Chemical element properties",
                requires_auth=True,
                rate_limit=100
            ),
            "materials_db": APIEndpoint(
                name="Materials Database",
                base_url="https://api.materialsproject.org",
                description="Materials science database",
                requires_auth=True,
                rate_limit=100
            ),
            "unit_conversion": APIEndpoint(
                name="Unit Conversion API",
                base_url="https://api.convertapi.com/v1",
                description="Unit conversion service",
                requires_auth=False,
                rate_limit=60
            )
        }
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Return plugin metadata"""
        return PLUGIN_INFO
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a web API tool"""
        try:
            tool_methods = {
                "fetch_weather": self.fetch_weather_data,
                "geocode_address": self.geocode_address,
                "lookup_element": self.lookup_chemical_element,
                "convert_units": self.convert_units_api,
                "validate_formula": self.validate_chemical_formula,
                "search_materials": self.search_materials_database,
                "get_exchange_rate": self.get_exchange_rate,
                "validate_coordinates": self.validate_coordinates,
                "reverse_geocode": self.reverse_geocode,
                "api_status": self.check_api_status,
                "cached_request": self.make_cached_request,
                "bulk_lookup": self.bulk_api_lookup,
                "mock_api_call": self.mock_api_call  # For testing
            }
            
            if tool_name not in tool_methods:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": list(tool_methods.keys())
                }
            
            result = tool_methods[tool_name](parameters)
            
            return {
                "success": True,
                "tool": tool_name,
                "result": result,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "parameters": parameters
            }
    
    def fetch_weather_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch weather data for a location (mock implementation)"""
        location = params["location"]
        api_key = params.get("api_key", "demo_key")
        units = params.get("units", "metric")
        
        # Mock weather data (in real implementation, would call actual API)
        mock_data = {
            "location": location,
            "temperature": 22.5,
            "humidity": 65,
            "pressure": 1013.25,
            "weather": "clear sky",
            "wind_speed": 3.2,
            "units": units,
            "timestamp": time.time()
        }
        
        return {
            "weather_data": mock_data,
            "source": "OpenWeatherMap API (mock)",
            "units": units,
            "coordinates": {"lat": 40.7128, "lon": -74.0060},  # Mock coordinates
            "api_calls_remaining": 995,
            "cache_hit": False
        }
    
    def geocode_address(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert address to coordinates (mock implementation)"""
        address = params["address"]
        country = params.get("country", "")
        
        # Mock geocoding result
        mock_result = {
            "address": address,
            "formatted_address": f"{address}, Mock City, Mock Country",
            "coordinates": {
                "latitude": 40.7128 + hash(address) % 100 / 1000,
                "longitude": -74.0060 + hash(address) % 100 / 1000
            },
            "confidence": 0.95,
            "address_components": {
                "street": "Mock Street",
                "city": "Mock City", 
                "state": "Mock State",
                "country": "Mock Country",
                "postal_code": "12345"
            }
        }
        
        return {
            "geocoding_result": mock_result,
            "source": "Nominatim API (mock)",
            "query": address,
            "response_time": 0.15
        }
    
    def lookup_chemical_element(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Look up chemical element properties (mock implementation)"""
        element = params["element"].title()
        
        # Mock element database
        mock_elements = {
            "H": {"name": "Hydrogen", "atomic_number": 1, "atomic_mass": 1.008, "group": 1},
            "He": {"name": "Helium", "atomic_number": 2, "atomic_mass": 4.003, "group": 18},
            "C": {"name": "Carbon", "atomic_number": 6, "atomic_mass": 12.011, "group": 14},
            "N": {"name": "Nitrogen", "atomic_number": 7, "atomic_mass": 14.007, "group": 15},
            "O": {"name": "Oxygen", "atomic_number": 8, "atomic_mass": 15.999, "group": 16},
            "Fe": {"name": "Iron", "atomic_number": 26, "atomic_mass": 55.845, "group": 8},
            "Au": {"name": "Gold", "atomic_number": 79, "atomic_mass": 196.967, "group": 11}
        }
        
        if element not in mock_elements:
            return {
                "error": f"Element '{element}' not found",
                "available_elements": list(mock_elements.keys())
            }
        
        element_data = mock_elements[element]
        
        return {
            "element_symbol": element,
            "element_data": element_data,
            "additional_properties": {
                "electron_configuration": f"Mock config for {element}",
                "electronegativity": 2.0 + hash(element) % 30 / 10,
                "melting_point": 1000 + hash(element) % 2000,  # Mock values
                "boiling_point": 2000 + hash(element) % 3000
            },
            "source": "Periodic Table API (mock)"
        }
    
    def convert_units_api(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert units using external API (mock implementation)"""
        value = params["value"]
        from_unit = params["from_unit"]
        to_unit = params["to_unit"]
        unit_type = params.get("unit_type", "auto")
        
        # Mock conversion factors (simplified)
        conversions = {
            ("m", "ft"): 3.28084,
            ("ft", "m"): 0.3048,
            ("kg", "lb"): 2.20462,
            ("lb", "kg"): 0.453592,
            ("C", "F"): lambda c: c * 9/5 + 32,
            ("F", "C"): lambda f: (f - 32) * 5/9,
            ("Pa", "psi"): 0.000145038,
            ("psi", "Pa"): 6894.76
        }
        
        conversion_key = (from_unit, to_unit)
        
        if conversion_key in conversions:
            factor = conversions[conversion_key]
            if callable(factor):
                converted_value = factor(value)
            else:
                converted_value = value * factor
        else:
            return {
                "error": f"Conversion from {from_unit} to {to_unit} not supported",
                "supported_conversions": list(conversions.keys())
            }
        
        return {
            "original_value": value,
            "from_unit": from_unit,
            "converted_value": round(converted_value, 6),
            "to_unit": to_unit,
            "conversion_factor": factor if not callable(factor) else "function",
            "unit_type": unit_type,
            "source": "Unit Conversion API (mock)"
        }
    
    def validate_chemical_formula(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate chemical formula using external chemistry API (mock)"""
        formula = params["formula"]
        
        # Simple validation (mock)
        import re
        pattern = r'^[A-Z][a-z]?[0-9]*([A-Z][a-z]?[0-9]*)*$'
        is_valid = bool(re.match(pattern, formula))
        
        if is_valid:
            # Mock molecular weight calculation
            elements = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
            mock_weight = sum(
                (12.0 if el == 'C' else 1.0 if el == 'H' else 16.0 if el == 'O' else 14.0) * 
                (int(count) if count else 1)
                for el, count in elements
            )
            
            return {
                "formula": formula,
                "is_valid": True,
                "molecular_weight": round(mock_weight, 3),
                "elements_found": elements,
                "validation_source": "Chemistry API (mock)",
                "additional_info": {
                    "formula_type": "molecular",
                    "charge": 0,
                    "structural_info": "Mock structural information"
                }
            }
        else:
            return {
                "formula": formula,
                "is_valid": False,
                "error": "Invalid chemical formula format",
                "suggestions": ["Check element symbols", "Verify parentheses", "Check charge notation"]
            }
    
    def search_materials_database(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search materials database (mock implementation)"""
        material_query = params["query"]
        properties = params.get("properties", ["all"])
        max_results = params.get("max_results", 10)
        
        # Mock materials database results
        mock_materials = [
            {
                "material_id": "mp-1234",
                "formula": "Fe2O3",
                "name": "Iron(III) oxide",
                "space_group": "R-3c",
                "density": 5.24,
                "band_gap": 2.1,
                "formation_energy": -8.25
            },
            {
                "material_id": "mp-5678", 
                "formula": "Al2O3",
                "name": "Aluminum oxide",
                "space_group": "R-3c",
                "density": 3.97,
                "band_gap": 8.8,
                "formation_energy": -16.82
            }
        ]
        
        # Filter by query (simple string matching)
        filtered_results = [
            mat for mat in mock_materials 
            if material_query.lower() in mat["formula"].lower() or 
               material_query.lower() in mat["name"].lower()
        ]
        
        return {
            "query": material_query,
            "total_results": len(filtered_results),
            "results": filtered_results[:max_results],
            "properties_included": properties,
            "database": "Materials Project (mock)",
            "search_time": 0.25,
            "api_version": "2.0"
        }
    
    def check_api_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check status of configured APIs"""
        endpoint_name = params.get("endpoint", "all")
        
        if endpoint_name != "all" and endpoint_name not in self.endpoints:
            return {
                "error": f"Unknown endpoint: {endpoint_name}",
                "available_endpoints": list(self.endpoints.keys())
            }
        
        status_results = {}
        
        endpoints_to_check = [endpoint_name] if endpoint_name != "all" else list(self.endpoints.keys())
        
        for ep_name in endpoints_to_check:
            endpoint = self.endpoints[ep_name]
            
            # Mock status check
            status_results[ep_name] = {
                "endpoint": endpoint.name,
                "base_url": endpoint.base_url,
                "status": "online",  # Mock status
                "response_time": 0.1 + hash(ep_name) % 50 / 1000,  # Mock response time
                "rate_limit_remaining": endpoint.rate_limit - (hash(ep_name) % 10),
                "requires_auth": endpoint.requires_auth,
                "last_checked": time.time()
            }
        
        return {
            "api_status": status_results,
            "overall_status": "all_online",
            "checked_at": time.time()
        }
    
    def make_cached_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make an API request with caching"""
        url = params["url"]
        request_params = params.get("params", {})
        force_refresh = params.get("force_refresh", False)
        
        # Check cache first
        if not force_refresh:
            cached_response = self.cache.get(url, request_params)
            if cached_response:
                return {
                    "response": cached_response,
                    "cached": True,
                    "cache_age": "< 1 hour",
                    "url": url,
                    "params": request_params
                }
        
        # Mock API call (in real implementation, would make actual HTTP request)
        mock_response = {
            "status": "success",
            "data": {"mock": "data", "timestamp": time.time()},
            "request_id": f"req_{hash(url) % 10000}"
        }
        
        # Cache the response
        self.cache.set(url, request_params, mock_response)
        
        return {
            "response": mock_response,
            "cached": False,
            "url": url,
            "params": request_params,
            "response_time": 0.2
        }
    
    def mock_api_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock API call for testing purposes"""
        endpoint = params.get("endpoint", "test")
        data = params.get("data", {})
        delay = params.get("delay", 0)
        
        if delay > 0:
            time.sleep(min(delay, 5))  # Max 5 second delay for safety
        
        return {
            "mock_response": {
                "endpoint": endpoint,
                "received_data": data,
                "timestamp": time.time(),
                "response_id": hash(str(data)) % 10000
            },
            "api_info": {
                "simulated_delay": delay,
                "endpoint_called": endpoint,
                "success": True
            }
        }
    
    def _check_rate_limit(self, endpoint_name: str) -> bool:
        """Simple rate limiting check"""
        now = time.time()
        minute_start = int(now // 60) * 60
        
        key = f"{endpoint_name}:{minute_start}"
        current_count = self.request_counts.get(key, 0)
        
        endpoint = self.endpoints.get(endpoint_name)
        if not endpoint:
            return True  # Allow unknown endpoints
        
        if current_count >= endpoint.rate_limit:
            return False
        
        self.request_counts[key] = current_count + 1
        return True


# Plugin instance for auto-discovery
plugin_instance = WebAPIPlugin()

def get_plugin():
    """Entry point for plugin discovery"""
    return plugin_instance

if __name__ == "__main__":
    # Test the web API plugin
    api_plugin = WebAPIPlugin()
    
    print("Web API Integration Plugin Test")
    print("=" * 40)
    
    # Test weather API
    result = api_plugin.execute_tool("fetch_weather", {
        "location": "New York",
        "units": "metric"
    })
    print(f"Weather API: {result}")
    
    # Test chemical element lookup
    result = api_plugin.execute_tool("lookup_element", {
        "element": "Fe"
    })
    print(f"Element lookup: {result}")
    
    # Test unit conversion
    result = api_plugin.execute_tool("convert_units", {
        "value": 100,
        "from_unit": "C",
        "to_unit": "F"
    })
    print(f"Unit conversion: {result}")
    
    # Test API status check
    result = api_plugin.execute_tool("api_status", {})
    print(f"API status: {result}")
