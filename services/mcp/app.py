import os
import json
import yaml
import logging
import math
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import hashlib

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
import requests
import numpy as np
from scipy import constants
try:
    from mendeleev import element
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    import pubchempy as pcp
    CHEMISTRY_LIBS_AVAILABLE = True
except ImportError:
    CHEMISTRY_LIBS_AVAILABLE = False
    logger.warning("Chemistry libraries not available. Some features will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/logs/mcp.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
api = FastAPI(
    title="Multi-Context Protocol (MCP) Server",
    description="Domain-specific knowledge servers for chemistry, mechanical, and materials",
    version="1.0.0"
)

# Models
class DomainQuery(BaseModel):
    query: str
    context: Dict[str, Any] = {}
    limit: int = 10

class DomainResponse(BaseModel):
    domain: str
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    query_time: float

class KnowledgeItem(BaseModel):
    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = {}
    domain: str
    source: Optional[str] = None
    tags: List[str] = []

# Chemistry-specific models
class MolecularWeightRequest(BaseModel):
    formula: str

class MolecularWeightResponse(BaseModel):
    formula: str
    molecular_weight: float
    composition: Dict[str, int]
    molar_mass_breakdown: Dict[str, float]

class pHCalculationRequest(BaseModel):
    calculation_type: str  # "concentration_to_ph" or "ph_to_concentration"
    value: float
    ion_type: str = "H+"  # "H+" or "OH-"

class pHCalculationResponse(BaseModel):
    input_type: str
    input_value: float
    result: float
    result_units: str
    calculation_details: Dict[str, Any]

class MolecularPropertyRequest(BaseModel):
    smiles: Optional[str] = None
    name: Optional[str] = None
    formula: Optional[str] = None

class MolecularPropertyResponse(BaseModel):
    molecule_info: Dict[str, Any]
    properties: Dict[str, Any]
    descriptors: Dict[str, Any]

# Mechanical engineering models
class BeamCalculationRequest(BaseModel):
    beam_type: str  # "simply_supported", "cantilever", "fixed_both_ends"
    length: float  # meters
    load: float  # Newtons
    load_type: str = "point_center"  # "point_center", "point_end", "distributed"
    moment_of_inertia: float  # m^4
    elastic_modulus: float  # Pa
    material_yield_strength: Optional[float] = None  # Pa

class BeamCalculationResponse(BaseModel):
    input_parameters: Dict[str, Any]
    max_stress: float  # Pa
    max_deflection: float  # m
    safety_factor: Optional[float] = None
    stress_location: str
    deflection_location: str
    calculations: Dict[str, Any]

# Materials science models  
class MaterialPropertyRequest(BaseModel):
    material: str
    properties: List[str] = ["density", "elastic_modulus", "yield_strength"]
    temperature: float = 293.15  # K

class MaterialPropertyResponse(BaseModel):
    material: str
    temperature: float
    properties: Dict[str, Any]
    metadata: Dict[str, Any]

class MCPService:
    """Main MCP service managing domain-specific knowledge"""
    
    def __init__(self):
        self.data_dir = Path("/data")
        self.domains = {
            "chemistry": ChemistryDomain(),
            "mechanical": MechanicalDomain(), 
            "materials": MaterialsDomain()
        }
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)
        
        # Load domain data
        self._load_domain_data()
    
    def _load_domain_data(self):
        """Load data for all domains"""
        for domain_name, domain_obj in self.domains.items():
            domain_data_path = self.data_dir / domain_name
            domain_data_path.mkdir(exist_ok=True)
            domain_obj.load_data(domain_data_path)
            logger.info(f"Loaded {domain_name} domain with {len(domain_obj.knowledge_items)} items")

class BaseDomain:
    """Base class for domain-specific knowledge servers"""
    
    def __init__(self, name: str):
        self.name = name
        self.knowledge_items = {}
        self.metadata = {
            "name": name,
            "loaded_items": 0,
            "last_updated": None
        }
    
    def load_data(self, data_path: Path):
        """Load domain-specific data from files"""
        self.knowledge_items = {}
        
        # Load JSON files
        for json_file in data_path.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                self._process_json_data(data, json_file.name)
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        # Load YAML files
        for yaml_file in data_path.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                self._process_yaml_data(data, yaml_file.name)
            except Exception as e:
                logger.error(f"Error loading {yaml_file}: {e}")
        
        # Load text files
        for txt_file in data_path.glob("*.txt"):
            try:
                with open(txt_file, 'r') as f:
                    content = f.read()
                self._process_text_data(content, txt_file.name)
            except Exception as e:
                logger.error(f"Error loading {txt_file}: {e}")
        
        self.metadata["loaded_items"] = len(self.knowledge_items)
        self.metadata["last_updated"] = datetime.utcnow().isoformat()
    
    def _process_json_data(self, data: Any, filename: str):
        """Process JSON data - override in subclasses"""
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    self._add_knowledge_item(item, filename)
        elif isinstance(data, dict):
            self._add_knowledge_item(data, filename)
    
    def _process_yaml_data(self, data: Any, filename: str):
        """Process YAML data - override in subclasses"""
        self._process_json_data(data, filename)  # Same logic as JSON
    
    def _process_text_data(self, content: str, filename: str):
        """Process text data - override in subclasses"""
        item = {
            "title": filename,
            "content": content,
            "source": filename
        }
        self._add_knowledge_item(item, filename)
    
    def _add_knowledge_item(self, item_data: Dict[str, Any], source: str):
        """Add a knowledge item to the domain"""
        item_id = item_data.get('id') or hashlib.md5(
            (item_data.get('title', '') + source).encode()
        ).hexdigest()
        
        knowledge_item = KnowledgeItem(
            id=item_id,
            title=item_data.get('title', 'Untitled'),
            content=item_data.get('content', ''),
            metadata=item_data.get('metadata', {}),
            domain=self.name,
            source=source,
            tags=item_data.get('tags', [])
        )
        
        self.knowledge_items[item_id] = knowledge_item
    
    def query(self, query: str, context: Dict[str, Any] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Query domain knowledge - override for domain-specific logic"""
        query_lower = query.lower()
        results = []
        
        for item_id, item in self.knowledge_items.items():
            score = 0.0
            
            # Simple text matching
            if query_lower in item.title.lower():
                score += 2.0
            if query_lower in item.content.lower():
                score += 1.0
            
            # Tag matching
            for tag in item.tags:
                if query_lower in tag.lower():
                    score += 1.5
            
            if score > 0:
                results.append({
                    "id": item.id,
                    "title": item.title,
                    "content": item.content[:500] + "..." if len(item.content) > 500 else item.content,
                    "score": score,
                    "metadata": item.metadata,
                    "tags": item.tags
                })
        
        # Sort by score and limit results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

class ChemistryDomain(BaseDomain):
    """Chemistry-specific domain knowledge server with molecular calculations"""
    
    def __init__(self):
        super().__init__("chemistry")
        self.chemistry_keywords = [
            "molecule", "compound", "reaction", "catalyst", "pH", "concentration",
            "solvent", "polymer", "crystalline", "organic", "inorganic", "synthesis",
            "chemical", "formula", "element", "bond", "atomic", "molecular",
            "molarity", "stoichiometry", "thermodynamics", "kinetics", "equilibrium"
        ]
        
        # Element atomic masses (most common isotopes)
        self.atomic_masses = {
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
            'Sb': 121.760, 'Te': 127.60, 'I': 126.904, 'Xe': 131.293, 'Cs': 132.905,
            'Ba': 137.327, 'La': 138.905, 'Ce': 140.116, 'Pr': 140.908, 'Nd': 144.242,
            'Pm': 145.0, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25, 'Tb': 158.925,
            'Dy': 162.500, 'Ho': 164.930, 'Er': 167.259, 'Tm': 168.934, 'Yb': 173.054,
            'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.948, 'W': 183.84, 'Re': 186.207,
            'Os': 190.23, 'Ir': 192.217, 'Pt': 195.084, 'Au': 196.967, 'Hg': 200.59,
            'Tl': 204.383, 'Pb': 207.2, 'Bi': 208.980, 'Po': 209.0, 'At': 210.0,
            'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.038,
            'Pa': 231.036, 'U': 238.029
        }
    
    def query(self, query: str, context: Dict[str, Any] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Enhanced chemistry-specific query"""
        # First get base results
        results = super().query(query, context, limit * 2)
        
        # Apply chemistry-specific scoring
        query_lower = query.lower()
        for result in results:
            chemistry_bonus = 0.0
            
            # Boost score for chemistry keywords
            for keyword in self.chemistry_keywords:
                if keyword in query_lower:
                    chemistry_bonus += 0.5
                if keyword in result["title"].lower():
                    chemistry_bonus += 0.3
                if keyword in result["content"].lower():
                    chemistry_bonus += 0.2
            
            result["score"] += chemistry_bonus
        
        # Re-sort and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def parse_molecular_formula(self, formula: str) -> Dict[str, int]:
        """Parse a molecular formula and return element composition"""
        import re
        
        # Remove spaces and common grouping characters for simplification
        formula = formula.replace(' ', '').replace('(', '').replace(')', '')
        
        # Find all element-number pairs
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        
        composition = {}
        for element, count in matches:
            count = int(count) if count else 1
            composition[element] = composition.get(element, 0) + count
        
        return composition
    
    def calculate_molecular_weight(self, formula: str) -> Dict[str, Any]:
        """Calculate molecular weight from chemical formula"""
        try:
            composition = self.parse_molecular_formula(formula)
            
            total_weight = 0.0
            molar_mass_breakdown = {}
            
            for element, count in composition.items():
                if element not in self.atomic_masses:
                    raise ValueError(f"Unknown element: {element}")
                
                element_mass = self.atomic_masses[element] * count
                total_weight += element_mass
                molar_mass_breakdown[element] = element_mass
            
            return {
                "formula": formula,
                "molecular_weight": round(total_weight, 3),
                "composition": composition,
                "molar_mass_breakdown": molar_mass_breakdown
            }
            
        except Exception as e:
            raise ValueError(f"Error calculating molecular weight: {str(e)}")
    
    def calculate_ph(self, calculation_type: str, value: float, ion_type: str = "H+") -> Dict[str, Any]:
        """Calculate pH from concentration or vice versa"""
        try:
            if calculation_type == "concentration_to_ph":
                if ion_type == "H+":
                    if value <= 0:
                        raise ValueError("Concentration must be positive")
                    ph = -math.log10(value)
                    result_units = "pH"
                    calculation_details = {
                        "formula": "pH = -log10([H+])",
                        "concentration_M": value
                    }
                elif ion_type == "OH-":
                    if value <= 0:
                        raise ValueError("Concentration must be positive")
                    poh = -math.log10(value)
                    ph = 14 - poh
                    result_units = "pH"
                    calculation_details = {
                        "formula": "pOH = -log10([OH-]); pH = 14 - pOH",
                        "concentration_M": value,
                        "pOH": poh
                    }
                else:
                    raise ValueError("Invalid ion_type. Use 'H+' or 'OH-'")
                
                return {
                    "input_type": "concentration",
                    "input_value": value,
                    "result": round(ph, 2),
                    "result_units": result_units,
                    "calculation_details": calculation_details
                }
                
            elif calculation_type == "ph_to_concentration":
                if value < 0 or value > 14:
                    raise ValueError("pH must be between 0 and 14")
                
                if ion_type == "H+":
                    concentration = 10**(-value)
                    result_units = "M (molarity)"
                    calculation_details = {
                        "formula": "[H+] = 10^(-pH)",
                        "pH": value
                    }
                elif ion_type == "OH-":
                    poh = 14 - value
                    concentration = 10**(-poh)
                    result_units = "M (molarity)"
                    calculation_details = {
                        "formula": "pOH = 14 - pH; [OH-] = 10^(-pOH)",
                        "pH": value,
                        "pOH": poh
                    }
                else:
                    raise ValueError("Invalid ion_type. Use 'H+' or 'OH-'")
                
                return {
                    "input_type": "pH",
                    "input_value": value,
                    "result": concentration,
                    "result_units": result_units,
                    "calculation_details": calculation_details
                }
            
            else:
                raise ValueError("Invalid calculation_type. Use 'concentration_to_ph' or 'ph_to_concentration'")
                
        except Exception as e:
            raise ValueError(f"Error in pH calculation: {str(e)}")
    
    def get_molecular_properties(self, smiles: str = None, name: str = None, formula: str = None) -> Dict[str, Any]:
        """Get molecular properties using RDKit and PubChem"""
        try:
            if not CHEMISTRY_LIBS_AVAILABLE:
                raise ValueError("Chemistry libraries not available")
            
            mol = None
            molecule_info = {}
            
            # Try to get molecule from different inputs
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                molecule_info["smiles"] = smiles
                molecule_info["input_type"] = "smiles"
            elif name:
                # Try to get SMILES from PubChem
                try:
                    compounds = pcp.get_compounds(name, 'name')
                    if compounds:
                        smiles = compounds[0].canonical_smiles
                        mol = Chem.MolFromSmiles(smiles)
                        molecule_info["smiles"] = smiles
                        molecule_info["name"] = name
                        molecule_info["pubchem_cid"] = compounds[0].cid
                        molecule_info["input_type"] = "name"
                except:
                    pass
            elif formula:
                # For formula, we can only do basic calculations
                mol_weight_data = self.calculate_molecular_weight(formula)
                return {
                    "molecule_info": {
                        "formula": formula,
                        "input_type": "formula",
                        "molecular_weight": mol_weight_data["molecular_weight"]
                    },
                    "properties": mol_weight_data,
                    "descriptors": {}
                }
            
            if mol is None:
                raise ValueError("Could not create molecule from input")
            
            # Calculate molecular descriptors
            descriptors = {
                "molecular_weight": round(Descriptors.MolWt(mol), 2),
                "logP": round(Descriptors.MolLogP(mol), 2),
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "num_rings": rdMolDescriptors.CalcNumRings(mol),
                "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "tpsa": round(Descriptors.TPSA(mol), 2),  # Topological Polar Surface Area
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "h_bond_donors": Descriptors.NumHDonors(mol),
                "h_bond_acceptors": Descriptors.NumHAcceptors(mol)
            }
            
            # Basic properties
            properties = {
                "molecular_formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
                "molecular_weight": descriptors["molecular_weight"],
                "exact_mass": round(Descriptors.ExactMolWt(mol), 4)
            }
            
            return {
                "molecule_info": molecule_info,
                "properties": properties,
                "descriptors": descriptors
            }
            
        except Exception as e:
            raise ValueError(f"Error calculating molecular properties: {str(e)}")

class MechanicalDomain(BaseDomain):
    """Mechanical engineering domain knowledge server with structural calculations"""
    
    def __init__(self):
        super().__init__("mechanical")
        self.mechanical_keywords = [
            "force", "stress", "strain", "torque", "pressure", "tension", "compression",
            "beam", "shaft", "gear", "bearing", "joint", "mechanism", "machine",
            "newton", "pascal", "engineering", "structural", "material", "strength",
            "statics", "dynamics", "vibration", "fatigue", "failure", "design"
        ]
    
    def query(self, query: str, context: Dict[str, Any] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Enhanced mechanical engineering query"""
        results = super().query(query, context, limit * 2)
        
        query_lower = query.lower()
        for result in results:
            mechanical_bonus = 0.0
            
            for keyword in self.mechanical_keywords:
                if keyword in query_lower:
                    mechanical_bonus += 0.5
                if keyword in result["title"].lower():
                    mechanical_bonus += 0.3
                if keyword in result["content"].lower():
                    mechanical_bonus += 0.2
            
            result["score"] += mechanical_bonus
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def calculate_beam_stress_deflection(self, beam_type: str, length: float, load: float, 
                                       load_type: str, moment_of_inertia: float, 
                                       elastic_modulus: float, material_yield_strength: float = None) -> Dict[str, Any]:
        """Calculate stress and deflection in beams under load"""
        try:
            # Validate inputs
            if length <= 0:
                raise ValueError("Length must be positive")
            if load <= 0:
                raise ValueError("Load must be positive")
            if moment_of_inertia <= 0:
                raise ValueError("Moment of inertia must be positive")
            if elastic_modulus <= 0:
                raise ValueError("Elastic modulus must be positive")
            
            calculations = {}
            
            # Calculate maximum moment and deflection based on beam type and loading
            if beam_type == "simply_supported":
                if load_type == "point_center":
                    # Point load at center of simply supported beam
                    max_moment = load * length / 4  # P*L/4
                    max_deflection = (load * length**3) / (48 * elastic_modulus * moment_of_inertia)  # PL³/48EI
                    stress_location = "center of beam"
                    deflection_location = "center of beam"
                    calculations = {
                        "max_moment_formula": "M = P*L/4",
                        "deflection_formula": "δ = P*L³/(48*E*I)"
                    }
                elif load_type == "distributed":
                    # Uniformly distributed load on simply supported beam
                    w = load / length  # Convert total load to distributed load
                    max_moment = w * length**2 / 8  # wL²/8
                    max_deflection = (5 * w * length**4) / (384 * elastic_modulus * moment_of_inertia)  # 5wL⁴/384EI
                    stress_location = "center of beam"
                    deflection_location = "center of beam"
                    calculations = {
                        "distributed_load_w": w,
                        "max_moment_formula": "M = w*L²/8",
                        "deflection_formula": "δ = 5*w*L⁴/(384*E*I)"
                    }
                else:
                    raise ValueError("Unsupported load type for simply supported beam")
                    
            elif beam_type == "cantilever":
                if load_type == "point_end":
                    # Point load at free end of cantilever
                    max_moment = load * length  # P*L
                    max_deflection = (load * length**3) / (3 * elastic_modulus * moment_of_inertia)  # PL³/3EI
                    stress_location = "fixed end"
                    deflection_location = "free end"
                    calculations = {
                        "max_moment_formula": "M = P*L",
                        "deflection_formula": "δ = P*L³/(3*E*I)"
                    }
                elif load_type == "distributed":
                    # Uniformly distributed load on cantilever
                    w = load / length
                    max_moment = w * length**2 / 2  # wL²/2
                    max_deflection = (w * length**4) / (8 * elastic_modulus * moment_of_inertia)  # wL⁴/8EI
                    stress_location = "fixed end"
                    deflection_location = "free end"
                    calculations = {
                        "distributed_load_w": w,
                        "max_moment_formula": "M = w*L²/2",
                        "deflection_formula": "δ = w*L⁴/(8*E*I)"
                    }
                else:
                    raise ValueError("Unsupported load type for cantilever beam")
                    
            elif beam_type == "fixed_both_ends":
                if load_type == "point_center":
                    # Point load at center of fixed-end beam
                    max_moment = load * length / 8  # P*L/8
                    max_deflection = (load * length**3) / (192 * elastic_modulus * moment_of_inertia)  # PL³/192EI
                    stress_location = "supports (fixed ends)"
                    deflection_location = "center of beam"
                    calculations = {
                        "max_moment_formula": "M = P*L/8",
                        "deflection_formula": "δ = P*L³/(192*E*I)"
                    }
                elif load_type == "distributed":
                    # Uniformly distributed load on fixed-end beam
                    w = load / length
                    max_moment = w * length**2 / 12  # wL²/12
                    max_deflection = (w * length**4) / (384 * elastic_modulus * moment_of_inertia)  # wL⁴/384EI
                    stress_location = "supports (fixed ends)"
                    deflection_location = "center of beam"
                    calculations = {
                        "distributed_load_w": w,
                        "max_moment_formula": "M = w*L²/12",
                        "deflection_formula": "δ = w*L⁴/(384*E*I)"
                    }
                else:
                    raise ValueError("Unsupported load type for fixed-end beam")
            else:
                raise ValueError("Unsupported beam type")
            
            # Calculate maximum stress (assuming rectangular cross-section at neutral axis)
            # For actual design, section modulus (I/c) would be used
            # Here we provide a simplified calculation
            max_stress = max_moment / (moment_of_inertia**(1/2))  # Simplified for demonstration
            
            # Calculate safety factor if yield strength is provided
            safety_factor = None
            if material_yield_strength and material_yield_strength > 0:
                safety_factor = material_yield_strength / max_stress
            
            # Store calculation details
            calculations.update({
                "max_moment": max_moment,
                "beam_type": beam_type,
                "load_type": load_type,
                "input_load": load,
                "beam_length": length,
                "elastic_modulus": elastic_modulus,
                "moment_of_inertia": moment_of_inertia
            })
            
            return {
                "input_parameters": {
                    "beam_type": beam_type,
                    "length": length,
                    "load": load,
                    "load_type": load_type,
                    "moment_of_inertia": moment_of_inertia,
                    "elastic_modulus": elastic_modulus,
                    "material_yield_strength": material_yield_strength
                },
                "max_stress": round(max_stress, 2),
                "max_deflection": round(max_deflection, 6),
                "safety_factor": round(safety_factor, 2) if safety_factor else None,
                "stress_location": stress_location,
                "deflection_location": deflection_location,
                "calculations": calculations
            }
            
        except Exception as e:
            raise ValueError(f"Error in beam calculation: {str(e)}")

class MaterialsDomain(BaseDomain):
    """Materials science domain knowledge server with properties database"""
    
    def __init__(self):
        super().__init__("materials")
        self.materials_keywords = [
            "steel", "aluminum", "composite", "ceramic", "polymer", "metal", "alloy",
            "hardness", "ductility", "brittleness", "elasticity", "plasticity",
            "microstructure", "grain", "phase", "crystal", "defect", "properties",
            "metallurgy", "corrosion", "wear", "fracture", "toughness", "modulus"
        ]
        
        # Materials properties database (simplified for demonstration)
        self.materials_database = {
            "steel": {
                "1018_steel": {
                    "density": 7870,  # kg/m³
                    "elastic_modulus": 200e9,  # Pa
                    "yield_strength": 370e6,  # Pa
                    "ultimate_strength": 440e6,  # Pa
                    "hardness": 126,  # HB
                    "thermal_conductivity": 51.9,  # W/m·K
                    "melting_point": 1811,  # K
                    "specific_heat": 486,  # J/kg·K
                    "poisson_ratio": 0.29
                },
                "4140_steel": {
                    "density": 7850,
                    "elastic_modulus": 205e9,
                    "yield_strength": 655e6,
                    "ultimate_strength": 1020e6,
                    "hardness": 302,
                    "thermal_conductivity": 42.6,
                    "melting_point": 1816,
                    "specific_heat": 475,
                    "poisson_ratio": 0.28
                }
            },
            "aluminum": {
                "6061_aluminum": {
                    "density": 2700,
                    "elastic_modulus": 68.9e9,
                    "yield_strength": 276e6,
                    "ultimate_strength": 310e6,
                    "hardness": 95,
                    "thermal_conductivity": 167,
                    "melting_point": 925,
                    "specific_heat": 896,
                    "poisson_ratio": 0.33
                },
                "7075_aluminum": {
                    "density": 2810,
                    "elastic_modulus": 71.7e9,
                    "yield_strength": 503e6,
                    "ultimate_strength": 572e6,
                    "hardness": 150,
                    "thermal_conductivity": 130,
                    "melting_point": 905,
                    "specific_heat": 960,
                    "poisson_ratio": 0.33
                }
            },
            "titanium": {
                "ti_6al_4v": {
                    "density": 4430,
                    "elastic_modulus": 113.8e9,
                    "yield_strength": 880e6,
                    "ultimate_strength": 950e6,
                    "hardness": 334,
                    "thermal_conductivity": 6.7,
                    "melting_point": 1878,
                    "specific_heat": 546,
                    "poisson_ratio": 0.32
                }
            },
            "copper": {
                "pure_copper": {
                    "density": 8960,
                    "elastic_modulus": 110e9,
                    "yield_strength": 33e6,
                    "ultimate_strength": 210e6,
                    "hardness": 87,
                    "thermal_conductivity": 401,
                    "melting_point": 1358,
                    "specific_heat": 385,
                    "poisson_ratio": 0.34
                }
            }
        }
    
    def query(self, query: str, context: Dict[str, Any] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Enhanced materials science query"""
        results = super().query(query, context, limit * 2)
        
        query_lower = query.lower()
        for result in results:
            materials_bonus = 0.0
            
            for keyword in self.materials_keywords:
                if keyword in query_lower:
                    materials_bonus += 0.5
                if keyword in result["title"].lower():
                    materials_bonus += 0.3
                if keyword in result["content"].lower():
                    materials_bonus += 0.2
            
            result["score"] += materials_bonus
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def lookup_material_properties(self, material: str, properties: List[str], temperature: float = 293.15) -> Dict[str, Any]:
        """Look up material properties from database"""
        try:
            material_lower = material.lower()
            found_material = None
            found_category = None
            
            # Search for material in database
            for category, materials in self.materials_database.items():
                for mat_name, mat_data in materials.items():
                    if (material_lower in mat_name.lower() or 
                        mat_name.lower() in material_lower or
                        material_lower in category):
                        found_material = mat_data
                        found_category = category
                        break
                if found_material:
                    break
            
            if not found_material:
                raise ValueError(f"Material '{material}' not found in database")
            
            # Extract requested properties
            result_properties = {}
            available_properties = found_material.keys()
            
            for prop in properties:
                prop_lower = prop.lower().replace(' ', '_')
                
                # Find matching property (flexible matching)
                matching_prop = None
                for available_prop in available_properties:
                    if (prop_lower == available_prop.lower() or 
                        prop_lower in available_prop.lower() or
                        available_prop.lower() in prop_lower):
                        matching_prop = available_prop
                        break
                
                if matching_prop:
                    value = found_material[matching_prop]
                    
                    # Add units based on property type
                    if "density" in matching_prop:
                        units = "kg/m³"
                    elif "modulus" in matching_prop:
                        units = "Pa"
                    elif "strength" in matching_prop:
                        units = "Pa"
                    elif "hardness" in matching_prop:
                        units = "HB"
                    elif "thermal_conductivity" in matching_prop:
                        units = "W/m·K"
                    elif "melting_point" in matching_prop:
                        units = "K"
                    elif "specific_heat" in matching_prop:
                        units = "J/kg·K"
                    elif "poisson" in matching_prop:
                        units = "dimensionless"
                    else:
                        units = "unknown"
                    
                    result_properties[matching_prop] = {
                        "value": value,
                        "units": units,
                        "temperature": temperature
                    }
                else:
                    result_properties[prop] = {
                        "value": None,
                        "error": "Property not available"
                    }
            
            return {
                "material": material,
                "category": found_category,
                "temperature": temperature,
                "properties": result_properties,
                "metadata": {
                    "available_properties": list(available_properties),
                    "database_name": "internal_materials_db",
                    "note": "Properties at room temperature unless specified"
                }
            }
            
        except Exception as e:
            raise ValueError(f"Error looking up material properties: {str(e)}")

# Global service instance
mcp_service = MCPService()

@api.on_event("startup")
async def startup_event():
    """Initialize MCP service on startup"""
    logger.info("MCP service initialized")

@api.get("/health")
async def health_check():
    """Health check endpoint"""
    domain_stats = {}
    for domain_name, domain_obj in mcp_service.domains.items():
        domain_stats[domain_name] = {
            "items": len(domain_obj.knowledge_items),
            "last_updated": domain_obj.metadata.get("last_updated")
        }
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "domains": domain_stats
    }

@api.get("/domains")
async def list_domains():
    """List available domains"""
    domains_info = []
    for domain_name, domain_obj in mcp_service.domains.items():
        domains_info.append({
            "name": domain_name,
            "items_count": len(domain_obj.knowledge_items),
            "metadata": domain_obj.metadata
        })
    
    return {"domains": domains_info}

@api.post("/chemistry/query", response_model=DomainResponse)
async def query_chemistry(request: DomainQuery):
    """Query chemistry domain"""
    start_time = datetime.utcnow()
    
    try:
        chemistry_domain = mcp_service.domains["chemistry"]
        results = chemistry_domain.query(request.query, request.context, request.limit)
        
        query_time = (datetime.utcnow() - start_time).total_seconds()
        
        return DomainResponse(
            domain="chemistry",
            results=results,
            metadata={
                "total_items": len(chemistry_domain.knowledge_items),
                "keywords_matched": sum(1 for kw in chemistry_domain.chemistry_keywords if kw in request.query.lower())
            },
            query_time=query_time
        )
        
    except Exception as e:
        logger.error(f"Chemistry query error: {e}")
        raise HTTPException(status_code=500, detail=f"Chemistry query failed: {str(e)}")

@api.post("/mechanical/query", response_model=DomainResponse)
async def query_mechanical(request: DomainQuery):
    """Query mechanical engineering domain"""
    start_time = datetime.utcnow()
    
    try:
        mechanical_domain = mcp_service.domains["mechanical"]
        results = mechanical_domain.query(request.query, request.context, request.limit)
        
        query_time = (datetime.utcnow() - start_time).total_seconds()
        
        return DomainResponse(
            domain="mechanical",
            results=results,
            metadata={
                "total_items": len(mechanical_domain.knowledge_items),
                "keywords_matched": sum(1 for kw in mechanical_domain.mechanical_keywords if kw in request.query.lower())
            },
            query_time=query_time
        )
        
    except Exception as e:
        logger.error(f"Mechanical query error: {e}")
        raise HTTPException(status_code=500, detail=f"Mechanical query failed: {str(e)}")

@api.post("/materials/query", response_model=DomainResponse)
async def query_materials(request: DomainQuery):
    """Query materials science domain"""
    start_time = datetime.utcnow()
    
    try:
        materials_domain = mcp_service.domains["materials"]
        results = materials_domain.query(request.query, request.context, request.limit)
        
        query_time = (datetime.utcnow() - start_time).total_seconds()
        
        return DomainResponse(
            domain="materials",
            results=results,
            metadata={
                "total_items": len(materials_domain.knowledge_items),
                "keywords_matched": sum(1 for kw in materials_domain.materials_keywords if kw in request.query.lower())
            },
            query_time=query_time
        )
        
    except Exception as e:
        logger.error(f"Materials query error: {e}")
        raise HTTPException(status_code=500, detail=f"Materials query failed: {str(e)}")

@api.post("/domains/{domain_name}/upload")
async def upload_domain_file(
    domain_name: str,
    file: UploadFile = File(...),
    metadata: str = Form("{}")
):
    """Upload a file to a specific domain"""
    if domain_name not in mcp_service.domains:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_name}' not found")
    
    try:
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata)
        except:
            metadata_dict = {}
        
        # Save file to domain directory
        domain_data_path = mcp_service.data_dir / domain_name
        file_path = domain_data_path / file.filename
        
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Reload domain data
        domain_obj = mcp_service.domains[domain_name]
        domain_obj.load_data(domain_data_path)
        
        logger.info(f"Uploaded {file.filename} to {domain_name} domain")
        
        return {
            "success": True,
            "message": f"File uploaded to {domain_name} domain",
            "filename": file.filename,
            "items_loaded": len(domain_obj.knowledge_items)
        }
        
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@api.get("/domains/{domain_name}/items")
async def get_domain_items(domain_name: str, limit: int = 50):
    """Get items from a specific domain"""
    if domain_name not in mcp_service.domains:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_name}' not found")
    
    domain_obj = mcp_service.domains[domain_name]
    items = list(domain_obj.knowledge_items.values())[:limit]
    
    return {
        "domain": domain_name,
        "items": [
            {
                "id": item.id,
                "title": item.title,
                "content": item.content[:200] + "..." if len(item.content) > 200 else item.content,
                "tags": item.tags,
                "source": item.source
            }
            for item in items
        ],
        "total_items": len(domain_obj.knowledge_items),
        "showing": len(items)
    }

@api.post("/domains/reload")
async def reload_domains():
    """Reload all domain data"""
    try:
        mcp_service._load_domain_data()
        
        domain_stats = {}
        for domain_name, domain_obj in mcp_service.domains.items():
            domain_stats[domain_name] = len(domain_obj.knowledge_items)
        
        return {
            "success": True,
            "message": "All domains reloaded successfully",
            "domain_stats": domain_stats
        }
        
    except Exception as e:
        logger.error(f"Domain reload error: {e}")
        raise HTTPException(status_code=500, detail=f"Domain reload failed: {str(e)}")

# Chemistry calculation endpoints
@api.post("/chemistry/molecular_weight", response_model=MolecularWeightResponse)
async def calculate_molecular_weight(request: MolecularWeightRequest):
    """Calculate molecular weight from chemical formula"""
    try:
        chemistry_domain = mcp_service.domains["chemistry"]
        result = chemistry_domain.calculate_molecular_weight(request.formula)
        return MolecularWeightResponse(**result)
    except Exception as e:
        logger.error(f"Molecular weight calculation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@api.post("/chemistry/ph_calculation", response_model=pHCalculationResponse)
async def calculate_ph(request: pHCalculationRequest):
    """Calculate pH from concentration or vice versa"""
    try:
        chemistry_domain = mcp_service.domains["chemistry"]
        result = chemistry_domain.calculate_ph(
            request.calculation_type, 
            request.value, 
            request.ion_type
        )
        return pHCalculationResponse(**result)
    except Exception as e:
        logger.error(f"pH calculation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@api.post("/chemistry/molecular_properties", response_model=MolecularPropertyResponse)
async def get_molecular_properties(request: MolecularPropertyRequest):
    """Get molecular properties from SMILES, name, or formula"""
    try:
        chemistry_domain = mcp_service.domains["chemistry"]
        result = chemistry_domain.get_molecular_properties(
            smiles=request.smiles,
            name=request.name,
            formula=request.formula
        )
        return MolecularPropertyResponse(**result)
    except Exception as e:
        logger.error(f"Molecular properties error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Mechanical engineering calculation endpoints
@api.post("/mechanical/beam_calculation", response_model=BeamCalculationResponse)
async def calculate_beam_stress_deflection(request: BeamCalculationRequest):
    """Calculate stress and deflection in beams under load"""
    try:
        mechanical_domain = mcp_service.domains["mechanical"]
        result = mechanical_domain.calculate_beam_stress_deflection(
            beam_type=request.beam_type,
            length=request.length,
            load=request.load,
            load_type=request.load_type,
            moment_of_inertia=request.moment_of_inertia,
            elastic_modulus=request.elastic_modulus,
            material_yield_strength=request.material_yield_strength
        )
        return BeamCalculationResponse(**result)
    except Exception as e:
        logger.error(f"Beam calculation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Materials science endpoints
@api.post("/materials/properties", response_model=MaterialPropertyResponse)
async def lookup_material_properties(request: MaterialPropertyRequest):
    """Look up material properties from database"""
    try:
        materials_domain = mcp_service.domains["materials"]
        result = materials_domain.lookup_material_properties(
            material=request.material,
            properties=request.properties,
            temperature=request.temperature
        )
        return MaterialPropertyResponse(**result)
    except Exception as e:
        logger.error(f"Material properties lookup error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@api.get("/materials/database")
async def get_materials_database():
    """Get available materials in the database"""
    materials_domain = mcp_service.domains["materials"]
    
    database_info = {}
    for category, materials in materials_domain.materials_database.items():
        database_info[category] = {
            "materials": list(materials.keys()),
            "sample_properties": list(next(iter(materials.values())).keys()) if materials else []
        }
    
    return {
        "database": database_info,
        "total_categories": len(database_info),
        "total_materials": sum(len(mats["materials"]) for mats in database_info.values())
    }

# Capabilities endpoints
@api.get("/capabilities")
async def get_capabilities():
    """Get available calculation capabilities"""
    return {
        "chemistry": {
            "molecular_weight": "Calculate molecular weight from chemical formula",
            "ph_calculation": "Convert between pH and concentration",
            "molecular_properties": "Get molecular descriptors from SMILES/name/formula",
            "libraries_available": CHEMISTRY_LIBS_AVAILABLE
        },
        "mechanical": {
            "beam_calculation": "Calculate stress and deflection in beams",
            "supported_beam_types": ["simply_supported", "cantilever", "fixed_both_ends"],
            "supported_load_types": ["point_center", "point_end", "distributed"]
        },
        "materials": {
            "property_lookup": "Look up material properties from database",
            "available_properties": [
                "density", "elastic_modulus", "yield_strength", "ultimate_strength",
                "hardness", "thermal_conductivity", "melting_point", "specific_heat"
            ]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)