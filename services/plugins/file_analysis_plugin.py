"""
File Analysis Plugin
Data file processing, analysis, and pattern extraction
"""

import os
import re
import json
import csv
import math
import statistics
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Plugin metadata
PLUGIN_INFO = {
    "name": "file_analysis",
    "version": "2.0.0",
    "description": "Comprehensive file analysis and data processing for scientific and engineering data",
    "author": "AI Stack Data Team",
    "category": "data_analysis",
    "domains": ["general", "chemistry", "mechanical", "materials"],
    "tags": ["file", "data", "analysis", "processing", "statistics", "parsing"],
    "requires": ["os", "re", "json", "csv", "math", "statistics", "pathlib"]
}

class FileType(Enum):
    TEXT = "text"
    CSV = "csv"
    JSON = "json"
    LOG = "log"
    DATA = "data"
    UNKNOWN = "unknown"

class DataType(Enum):
    NUMERIC = "numeric"
    TEXT = "text"
    MIXED = "mixed"
    TIMESTAMP = "timestamp"
    BOOLEAN = "boolean"

@dataclass
class FileInfo:
    """File metadata and basic information"""
    path: str
    name: str
    size: int
    extension: str
    file_type: FileType
    encoding: str
    line_count: int
    readable: bool

@dataclass
class DataSummary:
    """Statistical summary of numerical data"""
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    quartiles: List[float]
    outliers: List[float]

@dataclass
class PatternMatch:
    """Pattern matching result"""
    pattern: str
    matches: List[str]
    count: int
    line_numbers: List[int]
    context: List[str]

class FileAnalysisPlugin:
    """File analysis and data processing plugin"""
    
    def __init__(self):
        self.max_file_size = 100 * 1024 * 1024  # 100MB default
        self.max_lines_preview = 1000
        
        # Common scientific/engineering patterns
        self.predefined_patterns = {
            "scientific_notation": r'[-+]?[0-9]*\.?[0-9]+[eE][-+]?[0-9]+',
            "chemical_formula": r'[A-Z][a-z]?[0-9]*([A-Z][a-z]?[0-9]*)*',
            "temperature": r'\b\d+\.?\d*\s*[°]?[CFK]\b',
            "pressure": r'\b\d+\.?\d*\s*(Pa|bar|psi|atm|torr|mmHg)\b',
            "coordinates": r'[-+]?\d+\.?\d*\s*,\s*[-+]?\d+\.?\d*',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "url": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "date_iso": r'\d{4}-\d{2}-\d{2}',
            "time_24h": r'\d{2}:\d{2}:\d{2}',
            "numbers": r'[-+]?\d*\.?\d+',
            "words": r'\b[A-Za-z]+\b'
        }
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Return plugin metadata"""
        return PLUGIN_INFO
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a file analysis tool"""
        try:
            tool_methods = {
                "analyze_file": self.analyze_file,
                "extract_data": self.extract_numerical_data,
                "find_patterns": self.find_patterns,
                "statistics": self.calculate_statistics,
                "compare_files": self.compare_files,
                "clean_data": self.clean_data,
                "parse_csv": self.parse_csv_file,
                "parse_log": self.parse_log_file,
                "extract_metadata": self.extract_metadata,
                "validate_format": self.validate_file_format,
                "convert_format": self.convert_file_format,
                "merge_files": self.merge_files,
                "split_file": self.split_file,
                "filter_data": self.filter_data,
                "aggregate_data": self.aggregate_data,
                "detect_encoding": self.detect_file_encoding,
                "count_occurrences": self.count_occurrences
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
    
    def analyze_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive file analysis"""
        file_path = params["file_path"]
        include_preview = params.get("include_preview", True)
        max_preview_lines = params.get("max_preview_lines", 50)
        
        # Get basic file info
        file_info = self._get_file_info(file_path)
        
        if not file_info.readable:
            return {
                "file_info": file_info.__dict__,
                "error": "File is not readable or accessible"
            }
        
        # Read file content (with size limit)
        content = self._read_file_safe(file_path)
        
        # Detect data types and patterns
        data_analysis = self._analyze_content(content)
        
        # Generate preview
        preview = None
        if include_preview:
            lines = content.split('\n')
            preview_lines = min(max_preview_lines, len(lines))
            preview = {
                "first_lines": lines[:preview_lines//2],
                "last_lines": lines[-preview_lines//2:] if len(lines) > preview_lines else [],
                "total_lines": len(lines),
                "showing_lines": preview_lines
            }
        
        # Extract basic statistics
        numeric_data = self._extract_numbers(content)
        stats = None
        if numeric_data:
            stats = self._calculate_basic_stats(numeric_data)
        
        return {
            "file_info": file_info.__dict__,
            "content_analysis": data_analysis,
            "preview": preview,
            "statistics": stats,
            "file_health": self._assess_file_health(file_info, content)
        }
    
    def extract_numerical_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and analyze numerical data from file"""
        file_path = params["file_path"]
        column_separator = params.get("separator", None)
        specific_columns = params.get("columns", [])  # Column indices or names
        
        content = self._read_file_safe(file_path)
        
        if column_separator:
            # Structured data (CSV-like)
            data = self._extract_structured_numbers(content, column_separator, specific_columns)
        else:
            # Extract all numbers from text
            data = self._extract_numbers(content)
        
        if not data:
            return {"error": "No numerical data found in file"}
        
        # Calculate comprehensive statistics
        stats = self._calculate_comprehensive_stats(data)
        
        # Detect patterns in the data
        patterns = self._detect_numerical_patterns(data)
        
        return {
            "data_points": len(data),
            "data_sample": data[:10] if len(data) > 10 else data,
            "statistics": stats,
            "patterns": patterns,
            "data_quality": self._assess_data_quality(data)
        }
    
    def find_patterns(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Find patterns in file content"""
        file_path = params["file_path"]
        pattern_name = params.get("pattern_name")
        custom_pattern = params.get("custom_pattern")
        include_context = params.get("include_context", True)
        case_sensitive = params.get("case_sensitive", False)
        
        content = self._read_file_safe(file_path)
        
        results = {}
        
        if custom_pattern:
            # Use custom regex pattern
            results["custom"] = self._find_pattern_matches(
                content, custom_pattern, include_context, case_sensitive
            )
        
        if pattern_name:
            # Use predefined pattern
            if pattern_name in self.predefined_patterns:
                pattern = self.predefined_patterns[pattern_name]
                results[pattern_name] = self._find_pattern_matches(
                    content, pattern, include_context, case_sensitive
                )
            else:
                return {
                    "error": f"Unknown pattern: {pattern_name}",
                    "available_patterns": list(self.predefined_patterns.keys())
                }
        
        if not pattern_name and not custom_pattern:
            # Find all predefined patterns
            for name, pattern in self.predefined_patterns.items():
                matches = self._find_pattern_matches(content, pattern, False, case_sensitive)
                if matches["count"] > 0:
                    results[name] = matches
        
        return {
            "file_path": file_path,
            "patterns_found": results,
            "total_pattern_types": len(results),
            "search_settings": {
                "include_context": include_context,
                "case_sensitive": case_sensitive
            }
        }
    
    def calculate_statistics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical measures for numerical data in file"""
        file_path = params["file_path"]
        data_column = params.get("column", 0)  # For CSV files
        separator = params.get("separator", ",")
        
        content = self._read_file_safe(file_path)
        
        # Try to parse as structured data first
        try:
            if separator in content:
                numbers = self._extract_column_numbers(content, separator, data_column)
            else:
                numbers = self._extract_numbers(content)
        except:
            numbers = self._extract_numbers(content)
        
        if not numbers:
            return {"error": "No numerical data found"}
        
        # Calculate comprehensive statistics
        stats = self._calculate_comprehensive_stats(numbers)
        
        # Additional statistical tests
        distribution_analysis = self._analyze_distribution(numbers)
        
        return {
            "sample_size": len(numbers),
            "basic_statistics": stats,
            "distribution_analysis": distribution_analysis,
            "data_range": {
                "min": min(numbers),
                "max": max(numbers),
                "range": max(numbers) - min(numbers),
                "iqr": stats["q3"] - stats["q1"]
            },
            "quality_metrics": {
                "completeness": 1.0,  # No missing values in extracted numbers
                "uniqueness": len(set(numbers)) / len(numbers),
                "outlier_percentage": len(self._detect_outliers(numbers)) / len(numbers) * 100
            }
        }
    
    def parse_csv_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and analyze CSV file"""
        file_path = params["file_path"]
        delimiter = params.get("delimiter", ",")
        has_header = params.get("has_header", True)
        max_rows = params.get("max_rows", 1000)
        
        content = self._read_file_safe(file_path)
        lines = content.strip().split('\n')
        
        if not lines:
            return {"error": "File is empty"}
        
        # Parse CSV data
        parsed_data = []
        headers = None
        
        try:
            reader_lines = lines[:max_rows] if max_rows else lines
            
            for i, line in enumerate(reader_lines):
                row = [cell.strip('"') for cell in line.split(delimiter)]
                
                if i == 0 and has_header:
                    headers = row
                else:
                    parsed_data.append(row)
            
            # Analyze columns
            column_analysis = {}
            if parsed_data:
                num_columns = len(parsed_data[0])
                
                for col_idx in range(num_columns):
                    column_name = headers[col_idx] if headers else f"Column_{col_idx}"
                    column_data = [row[col_idx] if col_idx < len(row) else "" for row in parsed_data]
                    
                    # Analyze column data type and statistics
                    column_analysis[column_name] = self._analyze_column(column_data)
            
            return {
                "total_rows": len(lines),
                "parsed_rows": len(parsed_data),
                "headers": headers,
                "columns": len(headers) if headers else len(parsed_data[0]) if parsed_data else 0,
                "column_analysis": column_analysis,
                "sample_data": parsed_data[:5],  # First 5 rows
                "file_structure": {
                    "delimiter": delimiter,
                    "has_header": has_header,
                    "consistent_columns": self._check_column_consistency(parsed_data)
                }
            }
            
        except Exception as e:
            return {"error": f"Failed to parse CSV: {str(e)}"}
    
    def validate_file_format(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate file format and structure"""
        file_path = params["file_path"]
        expected_format = params.get("expected_format", "auto")
        validation_rules = params.get("rules", {})
        
        file_info = self._get_file_info(file_path)
        content = self._read_file_safe(file_path)
        
        validation_results = {
            "file_accessible": file_info.readable,
            "file_size_ok": file_info.size <= self.max_file_size,
            "encoding_valid": True,  # Simplified check
            "format_detection": self._detect_file_format(content, file_info.extension)
        }
        
        # Format-specific validation
        if expected_format == "csv" or validation_results["format_detection"] == FileType.CSV:
            validation_results["csv_validation"] = self._validate_csv_format(content, validation_rules)
        elif expected_format == "json" or validation_results["format_detection"] == FileType.JSON:
            validation_results["json_validation"] = self._validate_json_format(content, validation_rules)
        
        # Content validation
        if validation_rules:
            validation_results["content_validation"] = self._validate_content_rules(content, validation_rules)
        
        # Overall validity
        validation_results["is_valid"] = all([
            validation_results["file_accessible"],
            validation_results["file_size_ok"],
            validation_results["encoding_valid"]
        ])
        
        return {
            "file_path": file_path,
            "validation_results": validation_results,
            "recommendations": self._generate_validation_recommendations(validation_results)
        }
    
    def _get_file_info(self, file_path: str) -> FileInfo:
        """Get basic file information"""
        try:
            path_obj = Path(file_path)
            
            if not path_obj.exists():
                return FileInfo(
                    path=file_path, name="", size=0, extension="", 
                    file_type=FileType.UNKNOWN, encoding="", line_count=0, readable=False
                )
            
            stat = path_obj.stat()
            
            return FileInfo(
                path=file_path,
                name=path_obj.name,
                size=stat.st_size,
                extension=path_obj.suffix.lower(),
                file_type=self._detect_file_type(path_obj.suffix),
                encoding="utf-8",  # Simplified
                line_count=0,  # Will be calculated when reading
                readable=os.access(file_path, os.R_OK)
            )
            
        except Exception:
            return FileInfo(
                path=file_path, name="", size=0, extension="", 
                file_type=FileType.UNKNOWN, encoding="", line_count=0, readable=False
            )
    
    def _read_file_safe(self, file_path: str, max_size: int = None) -> str:
        """Safely read file with size limits"""
        max_size = max_size or self.max_file_size
        
        try:
            file_size = os.path.getsize(file_path)
            if file_size > max_size:
                raise ValueError(f"File too large: {file_size} bytes (max: {max_size})")
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
                
        except Exception as e:
            raise ValueError(f"Cannot read file: {str(e)}")
    
    def _extract_numbers(self, content: str) -> List[float]:
        """Extract all numerical values from text"""
        pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        matches = re.findall(pattern, content)
        
        numbers = []
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        
        return numbers
    
    def _calculate_comprehensive_stats(self, data: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive statistical measures"""
        if not data:
            return {}
        
        sorted_data = sorted(data)
        n = len(data)
        
        stats = {
            "count": n,
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "mode": statistics.mode(data) if len(set(data)) < len(data) else None,
            "std_dev": statistics.stdev(data) if n > 1 else 0,
            "variance": statistics.variance(data) if n > 1 else 0,
            "min": min(data),
            "max": max(data),
            "range": max(data) - min(data),
            "sum": sum(data)
        }
        
        # Quartiles
        stats["q1"] = sorted_data[n // 4] if n >= 4 else sorted_data[0]
        stats["q3"] = sorted_data[3 * n // 4] if n >= 4 else sorted_data[-1]
        stats["iqr"] = stats["q3"] - stats["q1"]
        
        # Additional measures
        if stats["mean"] != 0:
            stats["coefficient_of_variation"] = stats["std_dev"] / abs(stats["mean"])
        
        return stats
    
    def _find_pattern_matches(self, content: str, pattern: str, include_context: bool, case_sensitive: bool) -> Dict[str, Any]:
        """Find all matches for a regex pattern"""
        flags = 0 if case_sensitive else re.IGNORECASE
        
        try:
            matches = re.finditer(pattern, content, flags)
            results = []
            line_numbers = []
            contexts = []
            
            lines = content.split('\n')
            
            for match in matches:
                results.append(match.group())
                
                if include_context:
                    # Find line number
                    start_pos = match.start()
                    line_start = content.rfind('\n', 0, start_pos) + 1
                    line_num = content[:start_pos].count('\n') + 1
                    line_numbers.append(line_num)
                    
                    # Get context (line containing the match)
                    if line_num <= len(lines):
                        contexts.append(lines[line_num - 1])
            
            return {
                "pattern": pattern,
                "matches": results[:100],  # Limit to first 100 matches
                "count": len(results),
                "line_numbers": line_numbers[:100],
                "context": contexts[:100]
            }
            
        except re.error as e:
            return {
                "error": f"Invalid regex pattern: {str(e)}",
                "pattern": pattern,
                "matches": [],
                "count": 0
            }
    
    def _detect_file_type(self, extension: str) -> FileType:
        """Detect file type from extension"""
        ext_map = {
            '.txt': FileType.TEXT,
            '.csv': FileType.CSV,
            '.json': FileType.JSON,
            '.log': FileType.LOG,
            '.dat': FileType.DATA,
            '.data': FileType.DATA
        }
        return ext_map.get(extension.lower(), FileType.UNKNOWN)
    
    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content characteristics"""
        lines = content.split('\n')
        
        return {
            "total_characters": len(content),
            "total_lines": len(lines),
            "average_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
            "empty_lines": sum(1 for line in lines if not line.strip()),
            "numeric_lines": sum(1 for line in lines if self._is_numeric_line(line)),
            "contains_headers": self._detect_headers(lines[:5]),
            "delimiter_analysis": self._analyze_delimiters(content),
            "character_encoding": "utf-8"  # Simplified
        }
    
    def _is_numeric_line(self, line: str) -> bool:
        """Check if line contains primarily numeric data"""
        numbers = re.findall(r'[-+]?\d*\.?\d+', line)
        return len(numbers) > 0 and len(''.join(numbers)) > len(line.strip()) * 0.5
    
    def _detect_headers(self, first_lines: List[str]) -> bool:
        """Simple header detection"""
        if not first_lines:
            return False
        
        first_line = first_lines[0].strip()
        if not first_line:
            return False
        
        # Check if first line contains mostly text (likely headers)
        return len(re.findall(r'[a-zA-Z]', first_line)) > len(re.findall(r'\d', first_line))
    
    def _analyze_delimiters(self, content: str) -> Dict[str, int]:
        """Analyze potential delimiters in content"""
        delimiters = [',', ';', '\t', '|', ' ']
        analysis = {}
        
        for delimiter in delimiters:
            count = content.count(delimiter)
            analysis[delimiter] = count
        
        return analysis
    
    def _assess_file_health(self, file_info: FileInfo, content: str) -> Dict[str, Any]:
        """Assess overall file health and quality"""
        health = {
            "accessibility": "good" if file_info.readable else "poor",
            "size": "good" if file_info.size < self.max_file_size else "poor",
            "structure": "unknown"
        }
        
        # Assess structure based on content
        if file_info.file_type == FileType.CSV:
            lines = content.split('\n')
            consistent_cols = self._check_column_consistency([line.split(',') for line in lines if line.strip()])
            health["structure"] = "good" if consistent_cols else "poor"
        
        return health


# Plugin instance for auto-discovery
plugin_instance = FileAnalysisPlugin()

def get_plugin():
    """Entry point for plugin discovery"""
    return plugin_instance

if __name__ == "__main__":
    # Test the file analysis plugin
    analyzer = FileAnalysisPlugin()
    
    print("File Analysis Plugin Test")
    print("=" * 40)
    
    # Create a test file
    test_data = """Temperature,Pressure,Flow_Rate
    25.5,101.3,150.2
    26.1,102.1,148.7
    24.8,100.9,151.3
    """
    
    with open("/tmp/test_data.csv", "w") as f:
        f.write(test_data)
    
    # Test file analysis
    result = analyzer.execute_tool("analyze_file", {
        "file_path": "/tmp/test_data.csv"
    })
    print(f"File analysis: {result}")
    
    # Test pattern finding
    result = analyzer.execute_tool("find_patterns", {
        "file_path": "/tmp/test_data.csv",
        "pattern_name": "numbers"
    })
    print(f"Pattern analysis: {result}")
    
    # Clean up
    os.remove("/tmp/test_data.csv")
