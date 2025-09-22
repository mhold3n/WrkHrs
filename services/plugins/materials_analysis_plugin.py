"""
Materials Analysis Plugin
Advanced materials science calculations and property analysis
"""

import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Plugin metadata
PLUGIN_INFO = {
    "name": "materials_analysis",
    "version": "2.0.0",
    "description": "Advanced materials science analysis including stress, phase diagrams, and microstructure",
    "author": "AI Stack Materials Team", 
    "category": "materials",
    "domains": ["materials", "mechanical", "chemistry"],
    "tags": ["materials", "stress", "strain", "phase", "microstructure", "properties", "analysis"],
    "requires": ["math"]
}

class CrystalSystem(Enum):
    CUBIC = "cubic"
    TETRAGONAL = "tetragonal"
    ORTHORHOMBIC = "orthorhombic"
    HEXAGONAL = "hexagonal"
    TRIGONAL = "trigonal"
    MONOCLINIC = "monoclinic"
    TRICLINIC = "triclinic"

class MaterialClass(Enum):
    METAL = "metal"
    CERAMIC = "ceramic"
    POLYMER = "polymer"
    COMPOSITE = "composite"
    SEMICONDUCTOR = "semiconductor"

@dataclass
class MaterialProperties:
    """Material property data structure"""
    name: str
    density: float  # kg/m³
    elastic_modulus: float  # Pa
    yield_strength: float  # Pa
    ultimate_strength: float  # Pa
    poisson_ratio: float
    thermal_conductivity: float  # W/m·K
    thermal_expansion: float  # 1/K
    hardness: Optional[float] = None  # HV, HB, etc.
    melting_point: Optional[float] = None  # K
    specific_heat: Optional[float] = None  # J/kg·K

@dataclass
class StressState:
    """3D stress state representation"""
    sigma_x: float  # Pa
    sigma_y: float  # Pa
    sigma_z: float  # Pa
    tau_xy: float  # Pa
    tau_yz: float  # Pa
    tau_zx: float  # Pa

@dataclass
class StrainState:
    """3D strain state representation"""
    epsilon_x: float
    epsilon_y: float
    epsilon_z: float
    gamma_xy: float
    gamma_yz: float
    gamma_zx: float

class MaterialsAnalysisPlugin:
    """Advanced materials analysis and calculations"""
    
    def __init__(self):
        # Material database (simplified for demo)
        self.material_database = {
            "steel_1018": MaterialProperties(
                name="AISI 1018 Steel",
                density=7870,
                elastic_modulus=200e9,
                yield_strength=370e6,
                ultimate_strength=440e6,
                poisson_ratio=0.29,
                thermal_conductivity=51.9,
                thermal_expansion=12.1e-6,
                hardness=126,  # HB
                melting_point=1811,
                specific_heat=486
            ),
            "aluminum_6061": MaterialProperties(
                name="Aluminum 6061-T6",
                density=2700,
                elastic_modulus=68.9e9,
                yield_strength=276e6,
                ultimate_strength=310e6,
                poisson_ratio=0.33,
                thermal_conductivity=167,
                thermal_expansion=23.1e-6,
                hardness=95,  # HB
                melting_point=925,
                specific_heat=896
            ),
            "titanium_ti6al4v": MaterialProperties(
                name="Titanium Ti-6Al-4V",
                density=4430,
                elastic_modulus=113.8e9,
                yield_strength=880e6,
                ultimate_strength=950e6,
                poisson_ratio=0.32,
                thermal_conductivity=6.7,
                thermal_expansion=8.6e-6,
                hardness=334,  # HV
                melting_point=1878,
                specific_heat=546
            )
        }
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Return plugin metadata"""
        return PLUGIN_INFO
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a materials analysis tool"""
        try:
            tool_methods = {
                "stress_analysis": self.analyze_stress_state,
                "strain_analysis": self.analyze_strain_state,
                "von_mises_stress": self.calculate_von_mises_stress,
                "principal_stress": self.calculate_principal_stress,
                "safety_factor": self.calculate_safety_factor,
                "fatigue_life": self.estimate_fatigue_life,
                "thermal_stress": self.calculate_thermal_stress,
                "creep_analysis": self.analyze_creep,
                "fracture_toughness": self.calculate_fracture_toughness,
                "hardness_conversion": self.convert_hardness,
                "phase_diagram": self.analyze_phase_diagram,
                "grain_size": self.analyze_grain_size,
                "material_properties": self.lookup_material_properties,
                "composite_properties": self.calculate_composite_properties,
                "elastic_constants": self.calculate_elastic_constants,
                "thermal_properties": self.calculate_thermal_properties,
                "density_calculation": self.calculate_density,
                "porosity_analysis": self.analyze_porosity
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
    
    def analyze_stress_state(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze 3D stress state"""
        stress = StressState(
            sigma_x=params["sigma_x"],
            sigma_y=params["sigma_y"], 
            sigma_z=params["sigma_z"],
            tau_xy=params.get("tau_xy", 0),
            tau_yz=params.get("tau_yz", 0),
            tau_zx=params.get("tau_zx", 0)
        )
        
        # Calculate principal stresses
        principal_stresses = self._calculate_principal_stresses(stress)
        
        # Calculate von Mises stress
        von_mises = self._calculate_von_mises(stress)
        
        # Calculate maximum shear stress
        max_shear = (max(principal_stresses) - min(principal_stresses)) / 2
        
        # Calculate hydrostatic stress
        hydrostatic = (stress.sigma_x + stress.sigma_y + stress.sigma_z) / 3
        
        return {
            "input_stress_state": {
                "sigma_x": stress.sigma_x,
                "sigma_y": stress.sigma_y,
                "sigma_z": stress.sigma_z,
                "tau_xy": stress.tau_xy,
                "tau_yz": stress.tau_yz,
                "tau_zx": stress.tau_zx
            },
            "principal_stresses": {
                "sigma_1": round(principal_stresses[0], 2),
                "sigma_2": round(principal_stresses[1], 2),
                "sigma_3": round(principal_stresses[2], 2)
            },
            "von_mises_stress": round(von_mises, 2),
            "max_shear_stress": round(max_shear, 2),
            "hydrostatic_stress": round(hydrostatic, 2),
            "stress_invariants": {
                "I1": round(hydrostatic * 3, 2),
                "I2": round(self._stress_invariant_I2(stress), 2),
                "I3": round(self._stress_invariant_I3(stress), 2)
            }
        }
    
    def calculate_von_mises_stress(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate von Mises equivalent stress"""
        stress = StressState(
            sigma_x=params["sigma_x"],
            sigma_y=params["sigma_y"],
            sigma_z=params["sigma_z"],
            tau_xy=params.get("tau_xy", 0),
            tau_yz=params.get("tau_yz", 0),
            tau_zx=params.get("tau_zx", 0)
        )
        
        von_mises = self._calculate_von_mises(stress)
        
        return {
            "von_mises_stress": round(von_mises, 2),
            "stress_components": {
                "sigma_x": stress.sigma_x,
                "sigma_y": stress.sigma_y,
                "sigma_z": stress.sigma_z,
                "tau_xy": stress.tau_xy,
                "tau_yz": stress.tau_yz,
                "tau_zx": stress.tau_zx
            },
            "equation": "σ_vm = √[(σ₁-σ₂)² + (σ₂-σ₃)² + (σ₃-σ₁)²]/√2",
            "units": "Pa"
        }
    
    def calculate_safety_factor(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate safety factor based on failure criteria"""
        applied_stress = params["applied_stress"]  # Pa
        material_strength = params["material_strength"]  # Pa
        failure_theory = params.get("failure_theory", "von_mises")
        load_type = params.get("load_type", "static")
        
        if failure_theory == "von_mises":
            safety_factor = material_strength / applied_stress
        elif failure_theory == "maximum_stress":
            safety_factor = material_strength / applied_stress
        elif failure_theory == "tresca":
            # Maximum shear stress theory
            safety_factor = material_strength / (2 * applied_stress)
        else:
            raise ValueError(f"Unknown failure theory: {failure_theory}")
        
        # Apply factors for load type
        if load_type == "fatigue":
            # Rough fatigue factor (actual calculation is much more complex)
            safety_factor *= 0.5
        elif load_type == "impact":
            safety_factor *= 0.7
        
        # Safety assessment
        if safety_factor > 2.0:
            assessment = "Very Safe"
        elif safety_factor > 1.5:
            assessment = "Safe"
        elif safety_factor > 1.2:
            assessment = "Marginal"
        elif safety_factor > 1.0:
            assessment = "Unsafe"
        else:
            assessment = "Failure"
        
        return {
            "safety_factor": round(safety_factor, 3),
            "applied_stress": applied_stress,
            "material_strength": material_strength,
            "failure_theory": failure_theory,
            "load_type": load_type,
            "assessment": assessment,
            "recommendation": self._get_safety_recommendation(safety_factor)
        }
    
    def estimate_fatigue_life(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate fatigue life using S-N curve approximation"""
        stress_amplitude = params["stress_amplitude"]  # Pa
        ultimate_strength = params["ultimate_strength"]  # Pa
        yield_strength = params.get("yield_strength", ultimate_strength * 0.8)
        surface_finish = params.get("surface_finish", "machined")  # rough, machined, polished
        
        # Endurance limit (rough approximation)
        if ultimate_strength <= 1400e6:  # Steel
            endurance_limit = 0.5 * ultimate_strength
        else:  # High strength steel
            endurance_limit = 700e6
        
        # Surface finish factor
        surface_factors = {
            "polished": 1.0,
            "machined": 0.85,
            "hot_rolled": 0.75,
            "as_forged": 0.65,
            "rough": 0.5
        }
        surface_factor = surface_factors.get(surface_finish, 0.85)
        endurance_limit *= surface_factor
        
        # Basquin's equation parameters (rough estimates)
        fatigue_strength_coefficient = 1.5 * ultimate_strength
        fatigue_strength_exponent = -0.12  # Typical for steel
        
        if stress_amplitude <= endurance_limit:
            cycles_to_failure = float('inf')  # Infinite life
            life_category = "Infinite Life"
        else:
            # Basquin's equation: σ = σ'f * (2N)^b
            cycles_to_failure = ((stress_amplitude / fatigue_strength_coefficient) ** (1/fatigue_strength_exponent)) / 2
            
            if cycles_to_failure > 1e6:
                life_category = "High Cycle Fatigue"
            elif cycles_to_failure > 1e3:
                life_category = "Low Cycle Fatigue"
            else:
                life_category = "Ultra Low Cycle Fatigue"
        
        return {
            "cycles_to_failure": cycles_to_failure,
            "stress_amplitude": stress_amplitude,
            "endurance_limit": round(endurance_limit, 0),
            "surface_finish": surface_finish,
            "surface_factor": surface_factor,
            "life_category": life_category,
            "fatigue_strength_coefficient": fatigue_strength_coefficient,
            "fatigue_strength_exponent": fatigue_strength_exponent,
            "note": "This is a simplified estimation. Actual fatigue analysis requires detailed S-N curves and load history."
        }
    
    def calculate_thermal_stress(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate thermal stress due to temperature change"""
        temperature_change = params["temperature_change"]  # K
        thermal_expansion = params["thermal_expansion"]  # 1/K
        elastic_modulus = params["elastic_modulus"]  # Pa
        constraint = params.get("constraint", "fully_constrained")
        
        if constraint == "fully_constrained":
            thermal_stress = elastic_modulus * thermal_expansion * temperature_change
        elif constraint == "partially_constrained":
            constraint_factor = params.get("constraint_factor", 0.5)
            thermal_stress = constraint_factor * elastic_modulus * thermal_expansion * temperature_change
        else:  # free expansion
            thermal_stress = 0
        
        thermal_strain = thermal_expansion * temperature_change
        
        return {
            "thermal_stress": round(thermal_stress, 0),
            "thermal_strain": round(thermal_strain, 8),
            "temperature_change": temperature_change,
            "thermal_expansion": thermal_expansion,
            "elastic_modulus": elastic_modulus,
            "constraint": constraint,
            "equation": "σ_thermal = α * E * ΔT"
        }
    
    def calculate_composite_properties(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite material properties using rule of mixtures"""
        fiber_properties = params["fiber_properties"]  # Dict with E, density, etc.
        matrix_properties = params["matrix_properties"]  # Dict with E, density, etc.
        fiber_volume_fraction = params["fiber_volume_fraction"]
        orientation = params.get("orientation", "unidirectional")  # unidirectional, random, woven
        
        if fiber_volume_fraction < 0 or fiber_volume_fraction > 1:
            raise ValueError("Fiber volume fraction must be between 0 and 1")
        
        matrix_volume_fraction = 1 - fiber_volume_fraction
        
        # Rule of mixtures for different properties
        if orientation == "unidirectional":
            # Longitudinal modulus (Voigt model)
            E_longitudinal = (fiber_volume_fraction * fiber_properties["elastic_modulus"] + 
                            matrix_volume_fraction * matrix_properties["elastic_modulus"])
            
            # Transverse modulus (inverse rule of mixtures)
            E_transverse = 1 / (fiber_volume_fraction / fiber_properties["elastic_modulus"] + 
                              matrix_volume_fraction / matrix_properties["elastic_modulus"])
        else:
            # For random or woven, use modified rule of mixtures
            E_longitudinal = E_transverse = (fiber_volume_fraction * fiber_properties["elastic_modulus"] + 
                                           matrix_volume_fraction * matrix_properties["elastic_modulus"]) * 0.6
        
        # Density
        density = (fiber_volume_fraction * fiber_properties["density"] + 
                  matrix_volume_fraction * matrix_properties["density"])
        
        # Strength (simplified)
        strength = (fiber_volume_fraction * fiber_properties.get("strength", 0) + 
                   matrix_volume_fraction * matrix_properties.get("strength", 0))
        
        return {
            "longitudinal_modulus": round(E_longitudinal, 0),
            "transverse_modulus": round(E_transverse, 0),
            "density": round(density, 1),
            "strength": round(strength, 0),
            "fiber_volume_fraction": fiber_volume_fraction,
            "matrix_volume_fraction": matrix_volume_fraction,
            "orientation": orientation,
            "specific_modulus_longitudinal": round(E_longitudinal / density, 0),
            "specific_strength": round(strength / density, 0)
        }
    
    def analyze_grain_size(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze material properties based on grain size using Hall-Petch relation"""
        grain_size = params["grain_size"]  # μm
        base_yield_strength = params["base_yield_strength"]  # Pa
        hall_petch_constant = params.get("hall_petch_constant", 0.7e6)  # Pa·μm^0.5
        
        # Hall-Petch equation: σy = σ0 + k * d^(-1/2)
        grain_size_m = grain_size * 1e-6  # Convert to meters
        yield_strength = base_yield_strength + hall_petch_constant * (grain_size_m ** -0.5)
        
        # Grain size classification
        if grain_size < 1:
            grain_classification = "Ultra-fine"
        elif grain_size < 10:
            grain_classification = "Fine"
        elif grain_size < 100:
            grain_classification = "Medium"
        else:
            grain_classification = "Coarse"
        
        # ASTM grain size number (approximate)
        astm_grain_size = 1 - 3.32 * math.log10(grain_size / 1000)
        
        return {
            "grain_size_um": grain_size,
            "yield_strength": round(yield_strength, 0),
            "base_yield_strength": base_yield_strength,
            "hall_petch_constant": hall_petch_constant,
            "grain_classification": grain_classification,
            "astm_grain_size_number": round(astm_grain_size, 1),
            "equation": "σy = σ0 + k * d^(-1/2)",
            "strength_increase": round(yield_strength - base_yield_strength, 0)
        }
    
    def lookup_material_properties(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Look up material properties from database"""
        material_id = params["material_id"].lower()
        
        if material_id not in self.material_database:
            available = list(self.material_database.keys())
            return {
                "error": f"Material '{material_id}' not found",
                "available_materials": available
            }
        
        material = self.material_database[material_id]
        
        return {
            "material_name": material.name,
            "properties": {
                "density": material.density,
                "elastic_modulus": material.elastic_modulus,
                "yield_strength": material.yield_strength,
                "ultimate_strength": material.ultimate_strength,
                "poisson_ratio": material.poisson_ratio,
                "thermal_conductivity": material.thermal_conductivity,
                "thermal_expansion": material.thermal_expansion,
                "hardness": material.hardness,
                "melting_point": material.melting_point,
                "specific_heat": material.specific_heat
            },
            "units": {
                "density": "kg/m³",
                "elastic_modulus": "Pa",
                "yield_strength": "Pa",
                "ultimate_strength": "Pa",
                "poisson_ratio": "dimensionless",
                "thermal_conductivity": "W/m·K",
                "thermal_expansion": "1/K",
                "hardness": "HB/HV",
                "melting_point": "K",
                "specific_heat": "J/kg·K"
            }
        }
    
    def _calculate_principal_stresses(self, stress: StressState) -> List[float]:
        """Calculate principal stresses from stress state"""
        # For 3D case, solve cubic equation (simplified for demonstration)
        # This is a simplified calculation - actual implementation would use eigenvalue methods
        sigma_avg = (stress.sigma_x + stress.sigma_y + stress.sigma_z) / 3
        
        # Approximate principal stresses
        sigma_1 = max(stress.sigma_x, stress.sigma_y, stress.sigma_z)
        sigma_3 = min(stress.sigma_x, stress.sigma_y, stress.sigma_z)
        sigma_2 = stress.sigma_x + stress.sigma_y + stress.sigma_z - sigma_1 - sigma_3
        
        return sorted([sigma_1, sigma_2, sigma_3], reverse=True)
    
    def _calculate_von_mises(self, stress: StressState) -> float:
        """Calculate von Mises stress"""
        return math.sqrt(
            0.5 * ((stress.sigma_x - stress.sigma_y)**2 + 
                   (stress.sigma_y - stress.sigma_z)**2 + 
                   (stress.sigma_z - stress.sigma_x)**2) + 
            3 * (stress.tau_xy**2 + stress.tau_yz**2 + stress.tau_zx**2)
        )
    
    def _stress_invariant_I2(self, stress: StressState) -> float:
        """Calculate second stress invariant"""
        return (stress.sigma_x * stress.sigma_y + stress.sigma_y * stress.sigma_z + 
                stress.sigma_z * stress.sigma_x - stress.tau_xy**2 - 
                stress.tau_yz**2 - stress.tau_zx**2)
    
    def _stress_invariant_I3(self, stress: StressState) -> float:
        """Calculate third stress invariant (determinant)"""
        return (stress.sigma_x * stress.sigma_y * stress.sigma_z + 
                2 * stress.tau_xy * stress.tau_yz * stress.tau_zx - 
                stress.sigma_x * stress.tau_yz**2 - 
                stress.sigma_y * stress.tau_zx**2 - 
                stress.sigma_z * stress.tau_xy**2)
    
    def _get_safety_recommendation(self, safety_factor: float) -> str:
        """Get safety recommendation based on safety factor"""
        if safety_factor > 2.0:
            return "Design is very conservative. Consider optimization."
        elif safety_factor > 1.5:
            return "Design is safe for normal operation."
        elif safety_factor > 1.2:
            return "Design is marginal. Consider increasing safety factor."
        elif safety_factor > 1.0:
            return "Design is unsafe. Immediate redesign required."
        else:
            return "CRITICAL: Design will fail. Do not proceed."


# Plugin instance for auto-discovery
plugin_instance = MaterialsAnalysisPlugin()

def get_plugin():
    """Entry point for plugin discovery"""
    return plugin_instance

if __name__ == "__main__":
    # Test the materials analysis plugin
    materials = MaterialsAnalysisPlugin()
    
    print("Materials Analysis Plugin Test")
    print("=" * 40)
    
    # Test stress analysis
    result = materials.execute_tool("stress_analysis", {
        "sigma_x": 100e6,
        "sigma_y": 50e6,
        "sigma_z": 0,
        "tau_xy": 25e6
    })
    print(f"Stress analysis: {result}")
    
    # Test safety factor
    result = materials.execute_tool("safety_factor", {
        "applied_stress": 200e6,
        "material_strength": 400e6,
        "failure_theory": "von_mises"
    })
    print(f"Safety factor: {result}")
    
    # Test material lookup
    result = materials.execute_tool("material_properties", {
        "material_id": "steel_1018"
    })
    print(f"Material properties: {result}")
