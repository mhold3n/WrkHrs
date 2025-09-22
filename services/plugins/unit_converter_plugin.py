"""
Unit Converter Plugin - SI and Engineering Unit Conversions

This plugin provides comprehensive unit conversion functionality for engineering,
chemistry, and materials science applications with auto-discovery support.
"""

from typing import Dict, Any, List
from pluggy import HookspecMarker, HookimplMarker

# Plugin metadata
PLUGIN_INFO = {
    "name": "unit_converter",
    "version": "1.0.0",
    "description": "Comprehensive unit conversion for engineering and science",
    "author": "AI Stack Team",
    "category": "engineering",
    "domains": ["chemistry", "mechanical", "materials", "general"],
    "tags": ["units", "conversion", "SI", "engineering", "measurement"]
}

# Pluggy markers
hookspec = HookspecMarker("ai_stack")
hookimpl = HookimplMarker("ai_stack")

# Unit conversion factors (to base SI units)
UNIT_CONVERSIONS = {
    # Length (base: meter)
    "length": {
        "m": 1.0,
        "mm": 0.001,
        "cm": 0.01,
        "km": 1000.0,
        "in": 0.0254,
        "ft": 0.3048,
        "yd": 0.9144,
        "mil": 0.0000254,
        "μm": 1e-6,
        "nm": 1e-9,
        "Å": 1e-10  # Angstrom
    },
    
    # Mass (base: kilogram)
    "mass": {
        "kg": 1.0,
        "g": 0.001,
        "mg": 1e-6,
        "μg": 1e-9,
        "lb": 0.453592,
        "oz": 0.0283495,
        "ton": 1000.0,
        "amu": 1.66054e-27  # Atomic mass unit
    },
    
    # Time (base: second)
    "time": {
        "s": 1.0,
        "ms": 0.001,
        "μs": 1e-6,
        "ns": 1e-9,
        "min": 60.0,
        "hr": 3600.0,
        "day": 86400.0,
        "year": 31536000.0
    },
    
    # Temperature (special handling required)
    "temperature": {
        "K": {"offset": 0, "scale": 1},
        "°C": {"offset": 273.15, "scale": 1},
        "°F": {"offset": 459.67, "scale": 5/9},
        "°R": {"offset": 0, "scale": 5/9}  # Rankine
    },
    
    # Pressure (base: Pascal)
    "pressure": {
        "Pa": 1.0,
        "kPa": 1000.0,
        "MPa": 1e6,
        "GPa": 1e9,
        "bar": 1e5,
        "mbar": 100.0,
        "atm": 101325.0,
        "psi": 6894.76,
        "torr": 133.322,
        "mmHg": 133.322
    },
    
    # Energy (base: Joule)
    "energy": {
        "J": 1.0,
        "kJ": 1000.0,
        "MJ": 1e6,
        "cal": 4.184,
        "kcal": 4184.0,
        "eV": 1.60218e-19,
        "kWh": 3.6e6,
        "BTU": 1055.06
    },
    
    # Force (base: Newton)
    "force": {
        "N": 1.0,
        "kN": 1000.0,
        "MN": 1e6,
        "dyne": 1e-5,
        "lbf": 4.44822,
        "kgf": 9.80665
    },
    
    # Volume (base: cubic meter)
    "volume": {
        "m³": 1.0,
        "L": 0.001,
        "mL": 1e-6,
        "cm³": 1e-6,
        "mm³": 1e-9,
        "gal": 0.00378541,
        "qt": 0.000946353,
        "pt": 0.000473176,
        "fl_oz": 2.95735e-5
    },
    
    # Amount of substance (base: mole)
    "amount": {
        "mol": 1.0,
        "mmol": 0.001,
        "μmol": 1e-6,
        "nmol": 1e-9,
        "kmol": 1000.0
    }
}


class UnitConverterPlugin:
    """Unit converter plugin implementation"""
    
    @hookimpl
    def get_plugin_info(self) -> Dict[str, Any]:
        """Return plugin metadata"""
        return PLUGIN_INFO
    
    @hookimpl
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return list of available tools in this plugin"""
        return [
            {
                "name": "convert_length",
                "description": "Convert length units (m, mm, cm, km, in, ft, yd, mil, μm, nm, Å)",
                "parameters": {
                    "value": {"type": "float", "description": "Value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"}
                },
                "returns": {"type": "float", "description": "Converted value"}
            },
            {
                "name": "convert_mass",
                "description": "Convert mass units (kg, g, mg, μg, lb, oz, ton, amu)",
                "parameters": {
                    "value": {"type": "float", "description": "Value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"}
                },
                "returns": {"type": "float", "description": "Converted value"}
            },
            {
                "name": "convert_temperature",
                "description": "Convert temperature units (K, °C, °F, °R)",
                "parameters": {
                    "value": {"type": "float", "description": "Value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"}
                },
                "returns": {"type": "float", "description": "Converted value"}
            },
            {
                "name": "convert_pressure",
                "description": "Convert pressure units (Pa, kPa, MPa, GPa, bar, atm, psi, torr)",
                "parameters": {
                    "value": {"type": "float", "description": "Value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"}
                },
                "returns": {"type": "float", "description": "Converted value"}
            },
            {
                "name": "convert_energy",
                "description": "Convert energy units (J, kJ, MJ, cal, kcal, eV, kWh, BTU)",
                "parameters": {
                    "value": {"type": "float", "description": "Value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"}
                },
                "returns": {"type": "float", "description": "Converted value"}
            },
            {
                "name": "convert_units",
                "description": "General unit conversion with automatic unit type detection",
                "parameters": {
                    "value": {"type": "float", "description": "Value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"}
                },
                "returns": {"type": "object", "description": "Conversion result with metadata"}
            }
        ]
    
    @hookimpl
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool with given parameters"""
        try:
            value = float(parameters["value"])
            from_unit = parameters["from_unit"]
            to_unit = parameters["to_unit"]
            
            if tool_name == "convert_length":
                result = self.convert_length(value, from_unit, to_unit)
            elif tool_name == "convert_mass":
                result = self.convert_mass(value, from_unit, to_unit)
            elif tool_name == "convert_temperature":
                result = self.convert_temperature(value, from_unit, to_unit)
            elif tool_name == "convert_pressure":
                result = self.convert_pressure(value, from_unit, to_unit)
            elif tool_name == "convert_energy":
                result = self.convert_energy(value, from_unit, to_unit)
            elif tool_name == "convert_units":
                result = self.convert_units(value, from_unit, to_unit)
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
    
    def convert_length(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert length units"""
        return self._convert_simple_unit(value, from_unit, to_unit, "length")
    
    def convert_mass(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert mass units"""
        return self._convert_simple_unit(value, from_unit, to_unit, "mass")
    
    def convert_pressure(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert pressure units"""
        return self._convert_simple_unit(value, from_unit, to_unit, "pressure")
    
    def convert_energy(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert energy units"""
        return self._convert_simple_unit(value, from_unit, to_unit, "energy")
    
    def convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert temperature units"""
        temp_units = UNIT_CONVERSIONS["temperature"]
        
        if from_unit not in temp_units or to_unit not in temp_units:
            raise ValueError(f"Unsupported temperature unit. Supported: {list(temp_units.keys())}")
        
        # Convert to Kelvin first
        from_config = temp_units[from_unit]
        kelvin_value = (value + from_config["offset"]) * from_config["scale"]
        
        # Convert from Kelvin to target unit
        to_config = temp_units[to_unit]
        result = kelvin_value / to_config["scale"] - to_config["offset"]
        
        return result
    
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """General unit conversion with automatic type detection"""
        # Find which unit type both units belong to
        unit_type = None
        
        for category, units in UNIT_CONVERSIONS.items():
            if category == "temperature":
                if from_unit in units and to_unit in units:
                    unit_type = category
                    break
            else:
                if from_unit in units and to_unit in units:
                    unit_type = category
                    break
        
        if unit_type is None:
            raise ValueError(f"Cannot convert between {from_unit} and {to_unit} - incompatible unit types")
        
        # Perform conversion
        if unit_type == "temperature":
            result = self.convert_temperature(value, from_unit, to_unit)
        else:
            result = self._convert_simple_unit(value, from_unit, to_unit, unit_type)
        
        return {
            "value": result,
            "original_value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "unit_type": unit_type,
            "conversion_factor": result / value if value != 0 else None
        }
    
    def _convert_simple_unit(self, value: float, from_unit: str, to_unit: str, unit_type: str) -> float:
        """Convert simple units using multiplication factors"""
        units = UNIT_CONVERSIONS[unit_type]
        
        if from_unit not in units or to_unit not in units:
            raise ValueError(f"Unsupported {unit_type} unit. Supported: {list(units.keys())}")
        
        # Convert to base unit, then to target unit
        base_value = value * units[from_unit]
        result = base_value / units[to_unit]
        
        return result
    
    def get_supported_units(self, unit_type: str = None) -> Dict[str, List[str]]:
        """Get list of supported units by type"""
        if unit_type:
            if unit_type in UNIT_CONVERSIONS:
                return {unit_type: list(UNIT_CONVERSIONS[unit_type].keys())}
            else:
                raise ValueError(f"Unknown unit type: {unit_type}")
        
        return {category: list(units.keys()) for category, units in UNIT_CONVERSIONS.items()}


# Plugin instance for auto-discovery
plugin_instance = UnitConverterPlugin()


def get_plugin():
    """Entry point for plugin discovery"""
    return plugin_instance


# For direct usage
if __name__ == "__main__":
    converter = UnitConverterPlugin()
    
    # Test the converter
    print("Unit Converter Plugin Test")
    print("=" * 30)
    
    # Test length conversions
    print(f"1 m = {converter.convert_length(1, 'm', 'ft'):.4f} ft")
    print(f"100 mm = {converter.convert_length(100, 'mm', 'in'):.4f} in")
    print(f"1 Å = {converter.convert_length(1, 'Å', 'nm'):.4f} nm")
    
    # Test mass conversions
    print(f"1 kg = {converter.convert_mass(1, 'kg', 'lb'):.4f} lb")
    print(f"1 amu = {converter.convert_mass(1, 'amu', 'kg'):.6e} kg")
    
    # Test temperature conversions
    print(f"0°C = {converter.convert_temperature(0, '°C', 'K'):.2f} K")
    print(f"100°F = {converter.convert_temperature(100, '°F', '°C'):.2f}°C")
    
    # Test pressure conversions
    print(f"1 atm = {converter.convert_pressure(1, 'atm', 'Pa'):.0f} Pa")
    print(f"100 psi = {converter.convert_pressure(100, 'psi', 'MPa'):.4f} MPa")
    
    # Test general conversion
    print("\nTesting general conversion:")
    result = converter.convert_units(1000, "mm", "m")
    print(f"convert_units(1000, 'mm', 'm') = {result}")
    
    # Show supported units
    print(f"\nSupported unit types: {list(converter.get_supported_units().keys())}")