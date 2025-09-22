"""
Advanced Chemistry Toolkit Plugin
Comprehensive chemistry calculations and analysis tools for the AI Stack
"""

import math
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Plugin metadata
PLUGIN_INFO = {
    "name": "chemistry_toolkit",
    "version": "2.0.0", 
    "description": "Advanced chemistry calculations including thermodynamics, kinetics, and equilibrium",
    "author": "AI Stack Chemistry Team",
    "category": "chemistry",
    "domains": ["chemistry", "materials", "engineering"],
    "tags": ["chemistry", "thermodynamics", "kinetics", "equilibrium", "molecular", "stoichiometry"],
    "requires": ["math", "re"]
}

# Physical constants
class Constants:
    R = 8.314472  # Gas constant (J/mol·K)
    NA = 6.02214076e23  # Avogadro's number (1/mol)
    F = 96485.33212  # Faraday constant (C/mol)
    KB = 1.380649e-23  # Boltzmann constant (J/K)
    H = 6.62607015e-34  # Planck constant (J·s)
    C = 299792458  # Speed of light (m/s)

# Atomic masses (most common isotopes)
ATOMIC_MASSES = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
    'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
    'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
    'Ga': 69.723, 'Ge': 72.64, 'As': 74.922, 'Se': 78.96, 'Br': 79.904,
    'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224,
    'Nb': 92.906, 'Mo': 95.96, 'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.906,
    'Pd': 106.42, 'Ag': 107.868, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.710,
    'Sb': 121.760, 'Te': 127.60, 'I': 126.904, 'Xe': 131.293
}

@dataclass
class MolecularFormula:
    """Represents a parsed molecular formula"""
    formula: str
    elements: Dict[str, int]
    molecular_weight: float
    empirical_formula: str

@dataclass  
class ThermodynamicState:
    """Represents thermodynamic state variables"""
    temperature: float  # K
    pressure: float  # Pa
    volume: Optional[float] = None  # m³
    moles: Optional[float] = None  # mol
    enthalpy: Optional[float] = None  # J/mol
    entropy: Optional[float] = None  # J/mol·K
    gibbs_free_energy: Optional[float] = None  # J/mol

class ReactionType(Enum):
    SYNTHESIS = "synthesis"
    DECOMPOSITION = "decomposition" 
    SINGLE_REPLACEMENT = "single_replacement"
    DOUBLE_REPLACEMENT = "double_replacement"
    COMBUSTION = "combustion"
    ACID_BASE = "acid_base"
    REDOX = "redox"

class ChemistryToolkitPlugin:
    """Advanced chemistry toolkit with comprehensive calculations"""
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Return plugin metadata"""
        return PLUGIN_INFO
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a chemistry tool with given parameters"""
        try:
            # Map tool names to methods
            tool_methods = {
                "parse_formula": self.parse_molecular_formula,
                "molecular_weight": self.calculate_molecular_weight,
                "stoichiometry": self.calculate_stoichiometry,
                "ideal_gas": self.ideal_gas_law,
                "ph_buffer": self.calculate_buffer_ph,
                "equilibrium_constant": self.calculate_equilibrium_constant,
                "reaction_energy": self.calculate_reaction_energy,
                "arrhenius": self.arrhenius_equation,
                "nernst": self.nernst_equation,
                "concentration_units": self.convert_concentration_units,
                "molarity": self.calculate_molarity,
                "dilution": self.calculate_dilution,
                "gas_density": self.calculate_gas_density,
                "partial_pressure": self.calculate_partial_pressure,
                "vapor_pressure": self.antoine_equation,
                "reaction_quotient": self.calculate_reaction_quotient,
                "half_life": self.calculate_half_life,
                "rate_constant": self.calculate_rate_constant
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
    
    def parse_molecular_formula(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a molecular formula and return composition"""
        formula = params["formula"].replace(" ", "")
        
        # Parse formula using regex
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        
        elements = {}
        total_mass = 0.0
        
        for element, count in matches:
            count = int(count) if count else 1
            
            if element not in ATOMIC_MASSES:
                raise ValueError(f"Unknown element: {element}")
            
            elements[element] = elements.get(element, 0) + count
            total_mass += ATOMIC_MASSES[element] * count
        
        # Calculate empirical formula
        if elements:
            gcd_count = self._gcd_multiple(list(elements.values()))
            empirical_elements = {el: count // gcd_count for el, count in elements.items()}
            empirical_formula = "".join(
                f"{el}{count if count > 1 else ''}" 
                for el, count in sorted(empirical_elements.items())
            )
        else:
            empirical_formula = ""
        
        return {
            "formula": params["formula"],
            "elements": elements,
            "molecular_weight": round(total_mass, 3),
            "empirical_formula": empirical_formula,
            "atom_count": sum(elements.values()),
            "element_count": len(elements)
        }
    
    def calculate_molecular_weight(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate molecular weight from formula"""
        parsed = self.parse_molecular_formula(params)
        return {
            "formula": params["formula"],
            "molecular_weight": parsed["result"]["molecular_weight"],
            "units": "g/mol"
        }
    
    def calculate_stoichiometry(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate stoichiometric relationships"""
        reactant = params["reactant_formula"]
        product = params["product_formula"] 
        reactant_moles = params.get("reactant_moles", 1.0)
        coefficient_ratio = params.get("coefficient_ratio", [1, 1])  # [reactant_coeff, product_coeff]
        
        reactant_data = self.parse_molecular_formula({"formula": reactant})["result"]
        product_data = self.parse_molecular_formula({"formula": product})["result"]
        
        # Calculate molar ratio
        molar_ratio = coefficient_ratio[1] / coefficient_ratio[0]
        product_moles = reactant_moles * molar_ratio
        
        # Calculate masses
        reactant_mass = reactant_moles * reactant_data["molecular_weight"]
        product_mass = product_moles * product_data["molecular_weight"]
        
        return {
            "reactant": {
                "formula": reactant,
                "moles": reactant_moles,
                "mass_g": round(reactant_mass, 3),
                "molecular_weight": reactant_data["molecular_weight"]
            },
            "product": {
                "formula": product,
                "moles": round(product_moles, 6),
                "mass_g": round(product_mass, 3),
                "molecular_weight": product_data["molecular_weight"]
            },
            "molar_ratio": molar_ratio,
            "coefficient_ratio": coefficient_ratio
        }
    
    def ideal_gas_law(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate using ideal gas law PV = nRT"""
        # Extract known variables
        P = params.get("pressure")  # Pa
        V = params.get("volume")    # m³
        n = params.get("moles")     # mol
        T = params.get("temperature")  # K
        
        # Count known variables
        known_vars = sum(1 for var in [P, V, n, T] if var is not None)
        
        if known_vars != 3:
            raise ValueError("Exactly 3 variables must be provided to solve for the 4th")
        
        # Solve for unknown variable
        if P is None:
            P = (n * Constants.R * T) / V
            unknown = "pressure"
            result_units = "Pa"
        elif V is None:
            V = (n * Constants.R * T) / P
            unknown = "volume"
            result_units = "m³"
        elif n is None:
            n = (P * V) / (Constants.R * T)
            unknown = "moles"
            result_units = "mol"
        elif T is None:
            T = (P * V) / (n * Constants.R)
            unknown = "temperature"
            result_units = "K"
        
        return {
            "pressure": round(P, 6) if P else None,
            "volume": round(V, 6) if V else None,
            "moles": round(n, 6) if n else None,
            "temperature": round(T, 2) if T else None,
            "solved_for": unknown,
            "result_value": round(locals()[unknown[0].upper()], 6),
            "result_units": result_units,
            "gas_constant_used": Constants.R
        }
    
    def calculate_buffer_ph(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate pH of buffer solution using Henderson-Hasselbalch equation"""
        pka = params["pka"]
        acid_concentration = params["acid_concentration"]  # M
        base_concentration = params["base_concentration"]  # M
        
        # Henderson-Hasselbalch: pH = pKa + log([A-]/[HA])
        ratio = base_concentration / acid_concentration
        ph = pka + math.log10(ratio)
        
        # Calculate buffer capacity (simplified)
        total_concentration = acid_concentration + base_concentration
        buffer_capacity = 2.303 * total_concentration * (
            (acid_concentration * base_concentration) / (total_concentration ** 2)
        )
        
        return {
            "ph": round(ph, 2),
            "pka": pka,
            "acid_concentration": acid_concentration,
            "base_concentration": base_concentration,
            "concentration_ratio": round(ratio, 4),
            "buffer_capacity": round(buffer_capacity, 6),
            "total_concentration": total_concentration,
            "equation": "pH = pKa + log([A-]/[HA])"
        }
    
    def calculate_equilibrium_constant(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate equilibrium constant from concentrations"""
        products = params["products"]  # Dict[formula: concentration]
        reactants = params["reactants"]  # Dict[formula: concentration]
        coefficients = params.get("coefficients", {})  # Dict[formula: coefficient]
        temperature = params.get("temperature", 298.15)  # K
        
        # Calculate K = [products]^coeffs / [reactants]^coeffs
        k_numerator = 1.0
        k_denominator = 1.0
        
        for formula, concentration in products.items():
            coeff = coefficients.get(formula, 1)
            k_numerator *= concentration ** coeff
        
        for formula, concentration in reactants.items():
            coeff = coefficients.get(formula, 1)
            k_denominator *= concentration ** coeff
        
        k_eq = k_numerator / k_denominator
        
        return {
            "equilibrium_constant": round(k_eq, 8),
            "temperature": temperature,
            "products": products,
            "reactants": reactants,
            "coefficients": coefficients,
            "ln_k": round(math.log(k_eq), 4),
            "log_k": round(math.log10(k_eq), 4)
        }
    
    def arrhenius_equation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate using Arrhenius equation k = A * exp(-Ea/RT)"""
        # Can solve for k, A, Ea, or T given the other three
        k = params.get("rate_constant")
        A = params.get("pre_exponential_factor")
        Ea = params.get("activation_energy")  # J/mol
        T = params.get("temperature")  # K
        
        known_vars = sum(1 for var in [k, A, Ea, T] if var is not None)
        
        if known_vars != 3:
            raise ValueError("Exactly 3 variables must be provided")
        
        if k is None:
            k = A * math.exp(-Ea / (Constants.R * T))
            solved_for = "rate_constant"
        elif A is None:
            A = k / math.exp(-Ea / (Constants.R * T))
            solved_for = "pre_exponential_factor"
        elif Ea is None:
            Ea = -Constants.R * T * math.log(k / A)
            solved_for = "activation_energy"
        elif T is None:
            T = -Ea / (Constants.R * math.log(k / A))
            solved_for = "temperature"
        
        return {
            "rate_constant": k,
            "pre_exponential_factor": A,
            "activation_energy": Ea,
            "temperature": T,
            "solved_for": solved_for,
            "gas_constant": Constants.R,
            "equation": "k = A * exp(-Ea/RT)"
        }
    
    def nernst_equation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cell potential using Nernst equation"""
        E_standard = params["standard_potential"]  # V
        n = params["electrons_transferred"]
        temperature = params.get("temperature", 298.15)  # K
        Q = params.get("reaction_quotient", 1.0)
        
        # Nernst equation: E = E° - (RT/nF) * ln(Q)
        E_cell = E_standard - (Constants.R * temperature / (n * Constants.F)) * math.log(Q)
        
        # Calculate at standard conditions for comparison
        E_standard_calc = E_standard - (Constants.R * 298.15 / (n * Constants.F)) * math.log(Q)
        
        return {
            "cell_potential": round(E_cell, 4),
            "standard_potential": E_standard,
            "electrons_transferred": n,
            "temperature": temperature,
            "reaction_quotient": Q,
            "nernst_factor": round(Constants.R * temperature / (n * Constants.F), 6),
            "equation": "E = E° - (RT/nF) * ln(Q)",
            "faraday_constant": Constants.F
        }
    
    def convert_concentration_units(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert between different concentration units"""
        value = params["value"]
        from_unit = params["from_unit"].lower()
        to_unit = params["to_unit"].lower()
        molecular_weight = params.get("molecular_weight")  # g/mol
        density = params.get("density", 1.0)  # g/mL
        
        # Convert to molarity as intermediate
        if from_unit == "molarity":
            molarity = value
        elif from_unit == "molality":
            # Approximate conversion (assumes dilute solution)
            molarity = value * density
        elif from_unit == "mass_percent":
            if not molecular_weight:
                raise ValueError("Molecular weight required for mass percent conversion")
            # % w/w to molarity: (% * density * 10) / MW
            molarity = (value * density * 10) / molecular_weight
        elif from_unit == "ppm":
            if not molecular_weight:
                raise ValueError("Molecular weight required for ppm conversion")
            # ppm to molarity: (ppm * density) / (MW * 1000)
            molarity = (value * density) / (molecular_weight * 1000)
        else:
            raise ValueError(f"Unsupported from_unit: {from_unit}")
        
        # Convert from molarity to target unit
        if to_unit == "molarity":
            result = molarity
        elif to_unit == "molality":
            result = molarity / density
        elif to_unit == "mass_percent":
            if not molecular_weight:
                raise ValueError("Molecular weight required for mass percent conversion")
            result = (molarity * molecular_weight) / (density * 10)
        elif to_unit == "ppm":
            if not molecular_weight:
                raise ValueError("Molecular weight required for ppm conversion")
            result = (molarity * molecular_weight * 1000) / density
        else:
            raise ValueError(f"Unsupported to_unit: {to_unit}")
        
        return {
            "original_value": value,
            "from_unit": from_unit,
            "converted_value": round(result, 6),
            "to_unit": to_unit,
            "intermediate_molarity": round(molarity, 6),
            "molecular_weight": molecular_weight,
            "density": density
        }
    
    def calculate_molarity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate molarity from moles and volume"""
        moles = params.get("moles")
        volume_L = params.get("volume_L")
        mass_g = params.get("mass_g")
        molecular_weight = params.get("molecular_weight")
        
        # Calculate moles if not provided
        if moles is None:
            if mass_g is None or molecular_weight is None:
                raise ValueError("Either moles or (mass_g and molecular_weight) required")
            moles = mass_g / molecular_weight
        
        if volume_L is None:
            raise ValueError("Volume in liters is required")
        
        molarity = moles / volume_L
        
        return {
            "molarity": round(molarity, 6),
            "moles": round(moles, 6),
            "volume_L": volume_L,
            "mass_g": mass_g,
            "molecular_weight": molecular_weight,
            "units": "mol/L"
        }
    
    def calculate_dilution(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate dilution using C1V1 = C2V2"""
        C1 = params.get("initial_concentration")
        V1 = params.get("initial_volume")
        C2 = params.get("final_concentration")  
        V2 = params.get("final_volume")
        
        # Count known variables
        known = sum(1 for var in [C1, V1, C2, V2] if var is not None)
        
        if known != 3:
            raise ValueError("Exactly 3 variables must be provided")
        
        # Solve for unknown using C1V1 = C2V2
        if C1 is None:
            C1 = (C2 * V2) / V1
            solved_for = "initial_concentration"
        elif V1 is None:
            V1 = (C2 * V2) / C1
            solved_for = "initial_volume"
        elif C2 is None:
            C2 = (C1 * V1) / V2
            solved_for = "final_concentration"
        elif V2 is None:
            V2 = (C1 * V1) / C2
            solved_for = "final_volume"
        
        dilution_factor = C1 / C2 if C2 != 0 else float('inf')
        volume_added = V2 - V1 if V2 and V1 else None
        
        return {
            "initial_concentration": round(C1, 6),
            "initial_volume": round(V1, 6),
            "final_concentration": round(C2, 6),
            "final_volume": round(V2, 6),
            "dilution_factor": round(dilution_factor, 4),
            "volume_added": round(volume_added, 6) if volume_added else None,
            "solved_for": solved_for,
            "equation": "C1V1 = C2V2"
        }
    
    def _gcd_multiple(self, numbers: List[int]) -> int:
        """Calculate GCD of multiple numbers"""
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        result = numbers[0]
        for i in range(1, len(numbers)):
            result = gcd(result, numbers[i])
        return result


# Plugin instance for auto-discovery
plugin_instance = ChemistryToolkitPlugin()

def get_plugin():
    """Entry point for plugin discovery"""
    return plugin_instance

if __name__ == "__main__":
    # Test the chemistry toolkit
    toolkit = ChemistryToolkitPlugin()
    
    print("Chemistry Toolkit Plugin Test")
    print("=" * 40)
    
    # Test molecular formula parsing
    result = toolkit.execute_tool("parse_formula", {"formula": "C6H12O6"})
    print(f"Glucose formula: {result}")
    
    # Test ideal gas law
    result = toolkit.execute_tool("ideal_gas", {
        "pressure": 101325,  # Pa
        "volume": 0.0224,    # m³
        "temperature": 273.15  # K
    })
    print(f"Ideal gas (solve for moles): {result}")
    
    # Test buffer pH
    result = toolkit.execute_tool("ph_buffer", {
        "pka": 4.75,
        "acid_concentration": 0.1,
        "base_concentration": 0.1
    })
    print(f"Buffer pH: {result}")
