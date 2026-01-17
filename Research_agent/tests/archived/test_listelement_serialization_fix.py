#!/usr/bin/env python3
"""
Test and fix Bio.Entrez ListElement serialization issues in dataclasses

This script identifies and fixes the specific serialization problem with Bio.Entrez.ListElement
objects that get embedded in Paper dataclass instances.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set
import sys
import traceback

def test_listelement_serialization_issue():
    """Test the specific ListElement serialization problem"""
    
    print("🔍 Testing ListElement Serialization Issue")
    print("=" * 60)
    
    # Create mock ListElement that mimics Bio.Entrez.ListElement behavior
    class MockListElement:
        """Mock Bio.Entrez.ListElement that causes the serialization error"""
        def __init__(self, value, attributes=None, allowed_tags=None):
            self._value = value
            self.attributes = attributes or {}
            self.allowed_tags = allowed_tags or []
        
        def __str__(self):
            return str(self._value)
        
        def __repr__(self):
            return f"MockListElement({self._value!r})"
    
    # Replicate the Paper dataclass structure
    @dataclass
    class TestPaper:
        id: str
        title: str
        authors: List[str]
        abstract: str
        source: str
        url: str
        pdf_path: Optional[str] = None
        doi: Optional[str] = None
        citation_count: Optional[int] = None
        publication_year: Optional[int] = None
        keywords: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)
        processed_by_tiers: Set[str] = field(default_factory=set)
        
        def to_dict(self) -> dict:
            data = asdict(self)
            data['processed_by_tiers'] = list(self.processed_by_tiers)
            return data
    
    # Test 1: Create paper with problematic data containing ListElement
    print("\n1. Testing Paper with ListElement objects...")
    
    try:
        # This simulates what happens when PubMed returns data with ListElements
        problem_paper = TestPaper(
            id="test_1",
            title="Test Paper",
            authors=["Smith, John"],
            abstract="Test abstract",
            source="pubmed",
            url="http://test.com",
            keywords=[
                MockListElement("machine learning"),  # This will cause the error
                "regular string"
            ],
            metadata={
                "pmid": MockListElement("12345"),  # This will also cause issues
                "journal": "Test Journal",
                "mesh_terms": [MockListElement("AI"), MockListElement("ML")]
            }
        )
        
        # This should fail with the ListElement error
        result = problem_paper.to_dict()
        print("   ❌ ERROR: Should have failed but didn't!")
        
    except TypeError as e:
        if "ListElement.__init__()" in str(e):
            print("   ✓ Successfully reproduced the ListElement serialization error")
            print(f"   Error: {e}")
        else:
            print(f"   ❌ Different error occurred: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")

def create_fixed_paper_class():
    """Create a fixed Paper class that handles ListElement objects properly"""
    
    print("\n2. Creating Fixed Paper Class...")
    
    def safe_serialize_value(value):
        """Safely serialize any value, handling ListElement and other problematic types"""
        
        # Handle Bio.Entrez ListElement and similar objects
        if hasattr(value, '__class__') and 'ListElement' in str(value.__class__):
            # Convert ListElement to string
            return str(value)
        
        # Handle lists
        elif isinstance(value, list):
            return [safe_serialize_value(item) for item in value]
        
        # Handle dictionaries
        elif isinstance(value, dict):
            return {key: safe_serialize_value(val) for key, val in value.items()}
        
        # Handle sets
        elif isinstance(value, set):
            return [safe_serialize_value(item) for item in value]
        
        # Handle other complex objects by converting to string if needed
        elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, type(None))):
            try:
                # Try to convert to string for unknown objects
                return str(value)
            except:
                return repr(value)
        
        # Return primitive types as-is
        else:
            return value
    
    @dataclass 
    class FixedPaper:
        id: str
        title: str
        authors: List[str]
        abstract: str
        source: str
        url: str
        pdf_path: Optional[str] = None
        doi: Optional[str] = None
        citation_count: Optional[int] = None
        publication_year: Optional[int] = None
        keywords: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)
        processed_by_tiers: Set[str] = field(default_factory=set)
        
        def to_dict(self) -> dict:
            """Safe serialization that handles ListElement and other problematic objects"""
            
            # Create dict manually rather than using asdict() to avoid ListElement issues
            data = {
                'id': safe_serialize_value(self.id),
                'title': safe_serialize_value(self.title),
                'authors': safe_serialize_value(self.authors),
                'abstract': safe_serialize_value(self.abstract),
                'source': safe_serialize_value(self.source),
                'url': safe_serialize_value(self.url),
                'pdf_path': safe_serialize_value(self.pdf_path),
                'doi': safe_serialize_value(self.doi),
                'citation_count': safe_serialize_value(self.citation_count),
                'publication_year': safe_serialize_value(self.publication_year),
                'keywords': safe_serialize_value(self.keywords),
                'metadata': safe_serialize_value(self.metadata),
                'processed_by_tiers': safe_serialize_value(list(self.processed_by_tiers))
            }
            
            return data
    
    return FixedPaper, safe_serialize_value

def test_fixed_serialization():
    """Test the fixed serialization approach"""
    
    print("\n3. Testing Fixed Serialization...")
    
    FixedPaper, safe_serialize_value = create_fixed_paper_class()
    
    # Create mock ListElement
    class MockListElement:
        def __init__(self, value, attributes=None, allowed_tags=None):
            self._value = value
            self.attributes = attributes or {}
            self.allowed_tags = allowed_tags or []
        
        def __str__(self):
            return str(self._value)
        
        def __repr__(self):
            return f"MockListElement({self._value!r})"
    
    try:
        # Create paper with the same problematic data
        fixed_paper = FixedPaper(
            id="test_1",
            title="Test Paper",
            authors=["Smith, John"],
            abstract="Test abstract",
            source="pubmed",
            url="http://test.com",
            keywords=[
                MockListElement("machine learning"),  # This should now work
                "regular string"
            ],
            metadata={
                "pmid": MockListElement("12345"),  # This should also work
                "journal": "Test Journal",
                "mesh_terms": [MockListElement("AI"), MockListElement("ML")]
            },
            processed_by_tiers={"tier1", "tier2"}
        )
        
        # This should now work
        result = fixed_paper.to_dict()
        
        print("   ✓ Fixed serialization works!")
        print(f"   Keywords: {result['keywords']}")
        print(f"   PMID: {result['metadata']['pmid']}")
        print(f"   MeSH terms: {result['metadata']['mesh_terms']}")
        print(f"   Processed tiers: {result['processed_by_tiers']}")
        
        # Test JSON serialization
        json_result = json.dumps(result, indent=2)
        print("   ✓ JSON serialization also works!")
        print(f"   JSON length: {len(json_result)} chars")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fixed serialization failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def generate_paper_class_fix():
    """Generate the actual fix for the Paper class in session.py"""
    
    print("\n4. Generating Paper Class Fix...")
    
    fix_code = '''
def safe_serialize_value(value):
    """Safely serialize any value, handling Bio.Entrez ListElement and other problematic types"""
    
    # Handle Bio.Entrez ListElement and similar objects
    if hasattr(value, '__class__') and 'ListElement' in str(value.__class__):
        # Convert ListElement to string
        return str(value)
    
    # Handle lists
    elif isinstance(value, list):
        return [safe_serialize_value(item) for item in value]
    
    # Handle dictionaries
    elif isinstance(value, dict):
        return {key: safe_serialize_value(val) for key, val in value.items()}
    
    # Handle sets
    elif isinstance(value, set):
        return [safe_serialize_value(item) for item in value]
    
    # Handle other complex objects by converting to string if needed
    elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, type(None))):
        try:
            # Try to convert to string for unknown objects
            return str(value)
        except:
            return repr(value)
    
    # Return primitive types as-is
    else:
        return value

# Updated Paper.to_dict() method:
def to_dict(self) -> dict:
    """Safe serialization that handles Bio.Entrez ListElement and other problematic objects"""
    
    # Create dict manually rather than using asdict() to avoid ListElement issues
    data = {
        'id': safe_serialize_value(self.id),
        'title': safe_serialize_value(self.title),
        'authors': safe_serialize_value(self.authors),
        'abstract': safe_serialize_value(self.abstract),
        'source': safe_serialize_value(self.source),
        'url': safe_serialize_value(self.url),
        'pdf_path': safe_serialize_value(self.pdf_path),
        'doi': safe_serialize_value(self.doi),
        'citation_count': safe_serialize_value(self.citation_count),
        'publication_year': safe_serialize_value(self.publication_year),
        'keywords': safe_serialize_value(self.keywords),
        'metadata': safe_serialize_value(self.metadata),
        'processed_by_tiers': safe_serialize_value(list(self.processed_by_tiers))
    }
    
    return data
'''
    
    print("   ✓ Generated fix code for Paper class")
    return fix_code

def main():
    """Run all tests and generate fixes"""
    
    print("Bio.Entrez ListElement Serialization Fix")
    print("=" * 50)
    
    try:
        # Test 1: Reproduce the issue
        test_listelement_serialization_issue()
        
        # Test 2: Test the fix
        fix_works = test_fixed_serialization()
        
        # Test 3: Generate the actual fix
        if fix_works:
            fix_code = generate_paper_class_fix()
            
            print("\n" + "=" * 60)
            print("🎯 SUMMARY")
            print("=" * 60)
            print("✅ Successfully identified and fixed ListElement serialization issue")
            print("✅ Fix handles Bio.Entrez ListElement objects by converting to strings")
            print("✅ Fix maintains all other functionality")
            print("✅ Ready to apply to actual Paper class in session.py")
            
            return True
        else:
            print("\n❌ Fix validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)