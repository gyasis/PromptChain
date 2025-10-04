"""
Data Structure Validation Utilities

Provides validation functions to ensure data structure consistency across Research Agent components.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class DataStructureValidator:
    """
    Validates data structures for Research Agent components
    """
    
    @staticmethod
    def validate_paper_data(paper_data: Dict[str, Any], source: str = "unknown") -> Dict[str, Any]:
        """
        Validate and normalize paper data structure
        
        Args:
            paper_data: Paper data dictionary
            source: Source of the paper data (for logging)
            
        Returns:
            Validated and normalized paper data
        """
        if not isinstance(paper_data, dict):
            logger.error(f"Paper data from {source} is not a dictionary: {type(paper_data)}")
            return {}
        
        # Required fields for Paper class
        required_fields = {
            'id': str,
            'title': str, 
            'authors': list,
            'abstract': str,
            'source': str,
            'url': str
        }
        
        # Optional fields with defaults
        optional_fields = {
            'pdf_path': None,
            'doi': None,
            'citation_count': None,
            'publication_year': None,
            'keywords': [],
            'metadata': {},
            'processed_by_tiers': set()
        }
        
        validated_data = paper_data.copy()
        
        # Validate and fix required fields
        for field, expected_type in required_fields.items():
            if field not in validated_data:
                logger.warning(f"Missing required field '{field}' in paper from {source}")
                # Provide defaults
                if field == 'id':
                    validated_data[field] = f"unknown_{hash(str(paper_data))}"
                elif field == 'title':
                    validated_data[field] = 'Unknown Title'
                elif field == 'authors':
                    validated_data[field] = []
                elif field == 'abstract':
                    validated_data[field] = 'No abstract available'
                elif field == 'source':
                    validated_data[field] = source
                elif field == 'url':
                    validated_data[field] = ''
            elif not isinstance(validated_data[field], expected_type):
                logger.warning(f"Field '{field}' has incorrect type in paper from {source}: "
                             f"expected {expected_type}, got {type(validated_data[field])}")
                # Type conversion
                try:
                    if expected_type == list and not isinstance(validated_data[field], list):
                        if isinstance(validated_data[field], str):
                            validated_data[field] = [validated_data[field]]
                        else:
                            validated_data[field] = []
                    elif expected_type == str:
                        validated_data[field] = str(validated_data[field])
                except Exception as e:
                    logger.error(f"Failed to convert field '{field}': {e}")
                    validated_data[field] = required_fields[field]() if callable(required_fields[field]) else ""
        
        # Add optional fields with defaults
        for field, default in optional_fields.items():
            if field not in validated_data:
                if callable(default):
                    validated_data[field] = default()
                else:
                    validated_data[field] = default
        
        # Ensure metadata is a dictionary
        if not isinstance(validated_data.get('metadata'), dict):
            validated_data['metadata'] = {}
        
        # Move top-level fields that should be in metadata
        metadata_fields = ['journal', 'full_text_available', 'pdf_url', 'database_priority', 
                          'retrieved_at', 'search_term', 'search_method', 'arxiv_id', 
                          'pmid', 'categories', 'mesh_terms']
        
        for field in metadata_fields:
            if field in validated_data and field not in ['metadata']:
                validated_data['metadata'][field] = validated_data.pop(field)
        
        return validated_data
    
    @staticmethod
    def validate_query_data(query_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate and normalize query data structure
        
        Args:
            query_data: Query data (string or dictionary)
            
        Returns:
            Validated query dictionary
        """
        if isinstance(query_data, str):
            return {
                'text': query_data.strip(),
                'priority': 1.0,
                'category': 'general'
            }
        
        if not isinstance(query_data, dict):
            logger.error(f"Query data is not string or dictionary: {type(query_data)}")
            return {
                'text': str(query_data),
                'priority': 1.0,
                'category': 'general'
            }
        
        validated_query = query_data.copy()
        
        # Ensure required fields
        if 'text' not in validated_query:
            validated_query['text'] = 'Unknown query'
        
        if 'priority' not in validated_query:
            validated_query['priority'] = 1.0
        elif not isinstance(validated_query['priority'], (int, float)):
            try:
                validated_query['priority'] = float(validated_query['priority'])
            except:
                validated_query['priority'] = 1.0
        
        if 'category' not in validated_query:
            validated_query['category'] = 'general'
        
        # Ensure text is string
        if not isinstance(validated_query['text'], str):
            validated_query['text'] = str(validated_query['text'])
        
        return validated_query
    
    @staticmethod
    def validate_processing_result_data(result_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ProcessingResult data structure
        
        Args:
            result_data: Processing result data
            
        Returns:
            Validated result data
        """
        if not isinstance(result_data, dict):
            logger.error(f"Processing result data is not a dictionary: {type(result_data)}")
            return {}
        
        validated_data = result_data.copy()
        
        # Required fields
        required_fields = {
            'tier': str,
            'query_id': str,
            'paper_ids': list,
            'result_data': dict,
            'processing_time': (int, float)
        }
        
        for field, expected_type in required_fields.items():
            if field not in validated_data:
                logger.warning(f"Missing required field '{field}' in processing result")
                # Provide defaults
                if field == 'tier':
                    validated_data[field] = 'unknown'
                elif field == 'query_id':
                    validated_data[field] = 'unknown'
                elif field == 'paper_ids':
                    validated_data[field] = []
                elif field == 'result_data':
                    validated_data[field] = {}
                elif field == 'processing_time':
                    validated_data[field] = 0.0
            elif not isinstance(validated_data[field], expected_type):
                logger.warning(f"Field '{field}' has incorrect type in processing result: "
                             f"expected {expected_type}, got {type(validated_data[field])}")
                
                # Type conversion
                try:
                    if field == 'paper_ids' and not isinstance(validated_data[field], list):
                        # Convert single paper_id to list
                        if isinstance(validated_data[field], str):
                            validated_data[field] = [validated_data[field]]
                        else:
                            validated_data[field] = []
                    elif field == 'processing_time':
                        validated_data[field] = float(validated_data[field])
                    elif expected_type == str:
                        validated_data[field] = str(validated_data[field])
                    elif expected_type == dict and not isinstance(validated_data[field], dict):
                        validated_data[field] = {}
                except Exception as e:
                    logger.error(f"Failed to convert field '{field}': {e}")
                    # Use defaults
                    if field == 'tier':
                        validated_data[field] = 'unknown'
                    elif field == 'query_id':
                        validated_data[field] = 'unknown'
                    elif field == 'paper_ids':
                        validated_data[field] = []
                    elif field == 'result_data':
                        validated_data[field] = {}
                    elif field == 'processing_time':
                        validated_data[field] = 0.0
        
        return validated_data
    
    @staticmethod
    def validate_strategy_response(strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate search strategy response structure
        
        Args:
            strategy_data: Strategy response data
            
        Returns:
            Validated strategy data
        """
        if not isinstance(strategy_data, dict):
            logger.error(f"Strategy data is not a dictionary: {type(strategy_data)}")
            return DataStructureValidator._get_default_strategy()
        
        validated_data = strategy_data.copy()
        
        # Ensure required top-level sections exist
        required_sections = ['search_strategy', 'database_allocation', 'search_optimization']
        
        for section in required_sections:
            if section not in validated_data:
                logger.warning(f"Missing section '{section}' in strategy response")
                if section == 'search_strategy':
                    validated_data[section] = {
                        'primary_keywords': [],
                        'secondary_keywords': [],
                        'boolean_queries': []
                    }
                elif section == 'database_allocation':
                    validated_data[section] = DataStructureValidator._get_default_database_allocation()
                elif section == 'search_optimization':
                    validated_data[section] = {
                        'iteration_focus': 'General search strategy',
                        'gap_targeting': [],
                        'expansion_areas': [],
                        'exclusion_criteria': []
                    }
        
        return validated_data
    
    @staticmethod
    def _get_default_strategy() -> Dict[str, Any]:
        """Get default strategy structure"""
        return {
            'search_strategy': {
                'primary_keywords': [],
                'secondary_keywords': [],
                'boolean_queries': []
            },
            'database_allocation': DataStructureValidator._get_default_database_allocation(),
            'search_optimization': {
                'iteration_focus': 'General search strategy',
                'gap_targeting': [],
                'expansion_areas': [],
                'exclusion_criteria': []
            }
        }
    
    @staticmethod
    def _get_default_database_allocation() -> Dict[str, Any]:
        """Get default database allocation"""
        return {
            'sci_hub': {
                'priority': 1.0,
                'max_papers': 40,
                'search_terms': [],
                'rationale': 'Primary source for full papers'
            },
            'arxiv': {
                'priority': 0.8,
                'max_papers': 30,
                'search_terms': [],
                'rationale': 'Latest research and preprints'
            },
            'pubmed': {
                'priority': 0.6,
                'max_papers': 20,
                'search_terms': [],
                'rationale': 'Medical and biological research'
            }
        }


# Convenience functions for common validations
def validate_paper(paper_data: Dict[str, Any], source: str = "unknown") -> Dict[str, Any]:
    """Validate paper data - convenience wrapper"""
    return DataStructureValidator.validate_paper_data(paper_data, source)


def validate_query(query_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Validate query data - convenience wrapper"""
    return DataStructureValidator.validate_query_data(query_data)


def validate_processing_result(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate processing result - convenience wrapper"""
    return DataStructureValidator.validate_processing_result_data(result_data)


def validate_strategy(strategy_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate strategy data - convenience wrapper"""
    return DataStructureValidator.validate_strategy_response(strategy_data)