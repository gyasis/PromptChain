"""
Robust JSON Parser for LLM Responses

Handles common JSON parsing failures from LLM agents including:
- Mixed content responses (JSON + explanatory text)
- Multiple JSON objects in response
- Malformed JSON with missing brackets or quotes
- Empty/null responses
- Code blocks and markdown formatting
"""

import json
import re
import logging
from typing import Any, Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)


class RobustJSONParser:
    """
    A robust JSON parser designed to handle imperfect LLM responses
    """
    
    def __init__(self, 
                 strict_mode: bool = False,
                 fallback_enabled: bool = True,
                 max_depth: int = 10):
        """
        Initialize the robust JSON parser
        
        Args:
            strict_mode: If True, raises exceptions on parse failures
            fallback_enabled: If True, attempts fallback parsing on failures
            max_depth: Maximum nesting depth for JSON parsing safety
        """
        self.strict_mode = strict_mode
        self.fallback_enabled = fallback_enabled
        self.max_depth = max_depth
    
    def parse(self, 
              response: str, 
              expected_keys: Optional[List[str]] = None,
              fallback_structure: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse JSON from LLM response with robust error handling
        
        Args:
            response: Raw LLM response string
            expected_keys: List of keys that should exist in parsed JSON
            fallback_structure: Structure to return if all parsing fails
            
        Returns:
            Parsed JSON as dictionary
            
        Raises:
            JSONParseError: If strict_mode=True and parsing fails
        """
        # Pre-validation checks with enhanced logging
        if not response or not response.strip():
            logger.warning("🗝 EMPTY RESPONSE DETECTED 🗝")
            logger.warning(f"Response is: {repr(response)}")
            logger.warning(f"Response type: {type(response)}")
            logger.warning(f"Response length: {len(response) if response is not None else 'None'}")
            logger.warning("Using fallback structure for empty response")
            return fallback_structure or {}
        
        if len(response.strip()) < 3:  # Minimum for '{}'
            logger.warning("🗝 TOO SHORT RESPONSE DETECTED 🗝")
            logger.warning(f"Response length: {len(response)} chars")
            logger.warning(f"Response content: {repr(response)}")
            logger.warning("Using fallback structure for too-short response")
            return fallback_structure or {}
        
        # Special handling for common error patterns
        if self._is_malformed_response(response):
            logger.warning("Detected malformed response pattern - attempting repair")
            response = self._pre_process_malformed_response(response)
        
        # Step 1: Try direct JSON parsing
        try:
            parsed = self._direct_parse(response)
            if self._validate_structure(parsed, expected_keys):
                logger.debug("Direct parsing successful")
                return parsed
        except json.JSONDecodeError as e:
            if "Expecting value: line 1 column 1" in str(e):
                logger.warning("Empty JSON content detected - likely whitespace/formatting issue")
                response = self._clean_whitespace_issues(response)
                try:
                    parsed = self._direct_parse(response)
                    if self._validate_structure(parsed, expected_keys):
                        return parsed
                except:
                    pass
            logger.debug(f"Direct parsing failed: {e}")
        except Exception as e:
            logger.debug(f"Direct parsing failed: {e}")
        
        # Step 2: Try extracting JSON from mixed content
        try:
            parsed = self._extract_and_parse(response)
            if self._validate_structure(parsed, expected_keys):
                logger.debug("Extraction parsing successful")
                return parsed
        except Exception as e:
            logger.debug(f"Extraction parsing failed: {e}")
        
        # Step 3: Try multiple JSON object extraction
        try:
            parsed = self._parse_multiple_json(response)
            if self._validate_structure(parsed, expected_keys):
                logger.debug("Multiple JSON parsing successful")
                return parsed
        except Exception as e:
            logger.debug(f"Multiple JSON parsing failed: {e}")
        
        # Step 4: Try repair and parsing
        try:
            parsed = self._repair_and_parse(response)
            if self._validate_structure(parsed, expected_keys):
                logger.debug("Repair parsing successful")
                return parsed
        except Exception as e:
            logger.debug(f"Repair parsing failed: {e}")
        
        # Step 5: Fallback to structured extraction
        if self.fallback_enabled:
            logger.warning("All JSON parsing failed, using fallback extraction")
            result = self._fallback_extraction(response, expected_keys, fallback_structure)
            if result:
                return result
        
        # Final failure handling with comprehensive logging
        logger.error("🔥 ALL JSON PARSING METHODS FAILED 🔥")
        logger.error("=== COMPREHENSIVE PARSING FAILURE ANALYSIS ===")
        logger.error(f"Original response type: {type(response)}")
        logger.error(f"Original response length: {len(response)}")
        logger.error(f"Original response repr: {repr(response)}")
        logger.error(f"Expected keys: {expected_keys}")
        logger.error(f"Fallback structure provided: {fallback_structure is not None}")
        logger.error(f"Strict mode: {self.strict_mode}")
        
        # Check if response contains any meaningful content
        if response:
            non_whitespace_chars = len([c for c in response if not c.isspace()])
            json_chars = len([c for c in response if c in '{}[]":,'])
            alpha_chars = len([c for c in response if c.isalpha()])
            logger.error(f"Response analysis: non_whitespace={non_whitespace_chars}, json_chars={json_chars}, alpha_chars={alpha_chars}")
            
            # Check for common error patterns
            if 'error' in response.lower():
                logger.error("⚠️ RESPONSE CONTAINS 'ERROR' - LIKELY LLM ERROR RESPONSE")
            if 'sorry' in response.lower() or 'apologize' in response.lower():
                logger.error("⚠️ RESPONSE CONTAINS APOLOGY - LIKELY LLM REFUSAL")
            if len(response.strip()) == 0:
                logger.error("⚠️ RESPONSE IS EMPTY OR WHITESPACE ONLY")
            if response.count('{') == 0 and response.count('[') == 0:
                logger.error("⚠️ RESPONSE CONTAINS NO JSON STRUCTURAL CHARACTERS")
        
        logger.error("=== END COMPREHENSIVE PARSING FAILURE ANALYSIS ===")
        
        error_msg = f"Failed to parse JSON from response after all methods attempted"
        logger.error(error_msg)
        
        if self.strict_mode:
            raise JSONParseError(error_msg)
        
        return fallback_structure or {}
    
    def _direct_parse(self, response: str) -> Dict[str, Any]:
        """Try direct JSON parsing of the response"""
        response = response.strip()
        
        # Handle common wrapper cases
        for wrapper in ['```json\n', '```\n', '```']:
            if response.startswith(wrapper):
                response = response[len(wrapper):]
                break
        
        if response.endswith('```'):
            response = response[:-3]
        
        return json.loads(response)
    
    def _extract_and_parse(self, response: str) -> Dict[str, Any]:
        """Extract JSON from mixed content and parse"""
        
        # Method 1: Find balanced braces
        json_content = self._extract_balanced_json(response)
        if json_content:
            try:
                return json.loads(json_content)
            except:
                pass
        
        # Method 2: Regex extraction with different patterns
        patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested braces
            r'\{.*?\}',  # Simple braces (non-greedy)
            r'\{.*\}',   # Simple braces (greedy)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return parsed
                except:
                    continue
        
        # Method 3: Line-by-line assembly
        return self._assemble_from_lines(response)
    
    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """Extract JSON using balanced brace counting"""
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start_idx:i+1]
        
        return None
    
    def _parse_multiple_json(self, response: str) -> Dict[str, Any]:
        """Parse multiple JSON objects and merge or select best"""
        json_objects = []
        
        # Find all potential JSON objects
        lines = response.split('\n')
        current_json = ""
        brace_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            current_json += line + "\n"
            
            # Count braces
            for char in line:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    
                    if brace_count == 0 and current_json.strip():
                        try:
                            parsed = json.loads(current_json.strip())
                            if isinstance(parsed, dict):
                                json_objects.append(parsed)
                        except:
                            pass
                        current_json = ""
        
        if json_objects:
            # Return the most complete JSON object
            return max(json_objects, key=lambda x: len(x))
        
        raise ValueError("No valid JSON objects found")
    
    def _repair_and_parse(self, response: str) -> Dict[str, Any]:
        """Attempt to repair malformed JSON and parse"""
        
        # Common repairs
        repaired = response
        
        # Fix common issues
        repaired = re.sub(r',\s*}', '}', repaired)  # Remove trailing commas
        repaired = re.sub(r',\s*]', ']', repaired)  # Remove trailing commas in arrays
        repaired = re.sub(r'([{,]\s*)(\w+):', r'\1"\2":', repaired)  # Quote unquoted keys
        repaired = re.sub(r"'([^']*)'", r'"\1"', repaired)  # Replace single quotes
        
        # Try parsing the repaired version
        return json.loads(repaired)
    
    def _assemble_from_lines(self, response: str) -> Dict[str, Any]:
        """Assemble JSON from lines that might be split"""
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Filter for JSON-like lines
        json_lines = []
        for line in lines:
            if any(char in line for char in ['{', '}', '[', ']', ':', '"']):
                json_lines.append(line)
        
        if not json_lines:
            raise ValueError("No JSON-like content found")
        
        # Try to assemble
        assembled = '\n'.join(json_lines)
        return json.loads(assembled)
    
    def _validate_structure(self, 
                          parsed: Any, 
                          expected_keys: Optional[List[str]] = None) -> bool:
        """Validate that parsed JSON has expected structure"""
        if not isinstance(parsed, dict):
            return False
        
        if expected_keys:
            missing_keys = [key for key in expected_keys if key not in parsed]
            if missing_keys:
                logger.debug(f"Missing expected keys: {missing_keys}")
                return False
        
        return True
    
    def _fallback_extraction(self, 
                           response: str,
                           expected_keys: Optional[List[str]] = None,
                           fallback_structure: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract structured data using pattern matching as last resort"""
        
        if fallback_structure:
            return fallback_structure.copy()
        
        # Extract key-value pairs from text
        result = {}
        
        # Common patterns for query extraction
        if expected_keys and 'queries' in expected_keys:
            result['queries'] = self._extract_queries_from_text(response)
        
        if expected_keys and any(key in expected_keys for key in ['primary_queries', 'secondary_queries']):
            result.update(self._extract_query_sections_from_text(response))
        
        if expected_keys and 'search_strategy' in expected_keys:
            result['search_strategy'] = self._extract_search_strategy_from_text(response)
        
        if expected_keys and 'database_allocation' in expected_keys:
            result['database_allocation'] = self._extract_database_allocation_from_text(response)
        
        return result
    
    def _extract_queries_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract query-like structures from text"""
        queries = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if ('?' in line or 
                line.lower().startswith(('what', 'how', 'why', 'which', 'when', 'where')) or
                'query' in line.lower()):
                
                # Clean up the line
                clean_line = re.sub(r'^[•\-\*\d\.]+\s*', '', line)  # Remove bullets/numbers
                clean_line = clean_line.strip('"\'')
                
                if len(clean_line) > 10:  # Minimum meaningful length
                    queries.append({
                        'text': clean_line,
                        'priority': 1.0,
                        'category': 'general'
                    })
        
        return queries[:12]  # Limit to reasonable number
    
    def _extract_query_sections_from_text(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract queries organized by sections"""
        all_queries = self._extract_queries_from_text(text)
        
        # Distribute queries across sections
        total = len(all_queries)
        if total == 0:
            return {
                'primary_queries': [],
                'secondary_queries': [],
                'exploratory_queries': []
            }
        
        # Distribute roughly evenly
        primary_count = min(total // 3 + 1, 5)
        secondary_count = min((total - primary_count) // 2 + 1, 4)
        
        return {
            'primary_queries': all_queries[:primary_count],
            'secondary_queries': all_queries[primary_count:primary_count + secondary_count],
            'exploratory_queries': all_queries[primary_count + secondary_count:]
        }
    
    def _extract_search_strategy_from_text(self, text: str) -> Dict[str, Any]:
        """Extract search strategy from text"""
        # Extract keywords
        keywords = []
        for line in text.split('\n'):
            if 'keyword' in line.lower() or 'term' in line.lower():
                # Try to extract quoted strings or comma-separated values
                quoted = re.findall(r'"([^"]*)"', line)
                keywords.extend(quoted)
        
        return {
            'primary_keywords': keywords[:5],
            'secondary_keywords': keywords[5:10],
            'boolean_queries': [],
            'filters': {
                'publication_years': [2020, 2024],
                'paper_types': ['research'],
                'languages': ['en']
            }
        }
    
    def _extract_database_allocation_from_text(self, text: str) -> Dict[str, Any]:
        """Extract database allocation from text"""
        return {
            'sci_hub': {
                'priority': 1.0,
                'max_papers': 40,
                'search_terms': ['research'],
                'rationale': 'Primary source for full papers'
            },
            'arxiv': {
                'priority': 0.8,
                'max_papers': 30,
                'search_terms': ['preprints'],
                'rationale': 'Latest research methods'
            },
            'pubmed': {
                'priority': 0.6,
                'max_papers': 20,
                'search_terms': ['medical'],
                'rationale': 'Medical domain focus'
            }
        }
    
    def _is_malformed_response(self, response: str) -> bool:
        """Detect common malformed response patterns"""
        response = response.strip()
        
        # Check for line-split JSON (common LLM error)
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if len(lines) > 3:
            # Look for pattern: {, "key": [], "key": [], }
            if (lines[0] == '{' and 
                lines[-1] == '}' and
                all('":' in line or line in ['{', '}'] for line in lines)):
                return True
        
        # Check for completely empty content
        if not response or response.isspace():
            return True
        
        # Check for only punctuation/brackets
        if all(c in '{}[],":\'\n\t ' for c in response):
            return True
        
        return False
    
    def _pre_process_malformed_response(self, response: str) -> str:
        """Pre-process known malformed patterns"""
        response = response.strip()
        
        # Handle line-split JSON
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        if len(lines) > 2 and lines[0] == '{' and lines[-1] == '}':
            # Try to reconstruct JSON from split lines
            reconstructed = '{'
            for line in lines[1:-1]:
                if line.endswith(','):
                    reconstructed += line
                elif ':' in line and not line.endswith(','):
                    reconstructed += line + ','
                else:
                    reconstructed += line
            
            # Remove trailing comma and close
            reconstructed = reconstructed.rstrip(',') + '}'
            return reconstructed
        
        return response
    
    def _clean_whitespace_issues(self, response: str) -> str:
        """Clean common whitespace and formatting issues"""
        # Remove all whitespace and try to find JSON content
        cleaned = ''.join(response.split())
        
        # If still empty after cleaning, check for hidden characters
        if not cleaned:
            # Remove non-printable characters
            import re
            cleaned = re.sub(r'[^\x20-\x7E]', '', response).strip()
        
        # If still problematic, look for any JSON-like content
        if not cleaned or len(cleaned) < 2:
            # Extract any content between braces
            import re
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                cleaned = match.group(0)
            else:
                # Return minimal valid JSON
                cleaned = '{}'
        
        return cleaned


class JSONParseError(Exception):
    """Custom exception for JSON parsing failures"""
    pass


# Utility functions for common use cases
def parse_agent_response(response: str, 
                        agent_type: str = "generic",
                        strict: bool = False) -> Dict[str, Any]:
    """
    Convenience function for parsing agent responses
    
    Args:
        response: Raw agent response
        agent_type: Type of agent (query_generator, search_strategist, etc.)
        strict: Whether to use strict parsing mode
        
    Returns:
        Parsed response as dictionary
    """
    parser = RobustJSONParser(strict_mode=strict)
    
    # Define expected structures by agent type
    expected_keys = {
        'query_generator': ['primary_queries', 'secondary_queries', 'exploratory_queries'],
        'search_strategist': ['search_strategy', 'database_allocation'],
        'react_analyzer': ['gaps_identified', 'new_queries', 'should_continue'],
        'synthesis_agent': ['literature_review', 'key_findings', 'recommendations']
    }
    
    fallback_structures = {
        'query_generator': {
            'primary_queries': [],
            'secondary_queries': [],
            'exploratory_queries': []
        },
        'search_strategist': {
            'search_strategy': {
                'primary_keywords': [],
                'secondary_keywords': [],
                'boolean_queries': []
            },
            'database_allocation': {
                'sci_hub': {'priority': 1.0, 'max_papers': 40},
                'arxiv': {'priority': 0.8, 'max_papers': 30},
                'pubmed': {'priority': 0.6, 'max_papers': 20}
            }
        }
    }
    
    return parser.parse(
        response,
        expected_keys=expected_keys.get(agent_type),
        fallback_structure=fallback_structures.get(agent_type)
    )


def extract_json_from_mixed_content(text: str) -> Optional[Dict[str, Any]]:
    """
    Quick utility to extract JSON from mixed content
    
    Args:
        text: Text containing JSON mixed with other content
        
    Returns:
        Extracted JSON or None
    """
    parser = RobustJSONParser(strict_mode=False)
    try:
        return parser._extract_and_parse(text)
    except:
        return None


def validate_json_structure(data: Dict[str, Any], 
                          required_keys: List[str],
                          fix_missing: bool = True) -> Dict[str, Any]:
    """
    Validate and optionally fix JSON structure
    
    Args:
        data: JSON data to validate
        required_keys: Keys that must be present
        fix_missing: Whether to add missing keys with default values
        
    Returns:
        Validated/fixed JSON data
    """
    if not isinstance(data, dict):
        if fix_missing:
            data = {}
        else:
            raise ValueError("Data must be a dictionary")
    
    if fix_missing:
        for key in required_keys:
            if key not in data:
                # Add reasonable defaults based on key name
                if 'queries' in key:
                    data[key] = []
                elif 'strategy' in key:
                    data[key] = {}
                elif 'allocation' in key:
                    data[key] = {}
                else:
                    data[key] = None
    
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys and not fix_missing:
        raise ValueError(f"Missing required keys: {missing_keys}")
    
    return data