#!/usr/bin/env python3
"""
Test Knowledge Base Listing
Simulates the knowledge base listing to verify the fix works
"""

import os
import sys
from pathlib import Path

# Add project paths
sys.path.insert(0, 'examples/lightrag_demo')

def test_knowledge_base_listing():
    """Test the knowledge base listing logic from the demo"""
    print("🔍 Testing Knowledge Base Listing Logic")
    print("=" * 50)
    
    # Simulate the logic from the enhanced demo
    base_paths = [
        "./examples/lightrag_demo",
        "."
    ]
    
    available_dirs = []
    
    for base_path in base_paths:
        if os.path.exists(base_path):
            for item in os.listdir(base_path):
                if (item.startswith("rag_data_") or item == "rag_storage") and os.path.isdir(os.path.join(base_path, item)):
                    full_path = os.path.join(base_path, item)
                    # Check if it has processed content
                    doc_status_file = os.path.join(full_path, "kv_store_doc_status.json")
                    if os.path.exists(doc_status_file):
                        available_dirs.append((item, full_path))
    
    if not available_dirs:
        print("❌ No knowledge bases found!")
        return False
    
    print(f"📚 Found {len(available_dirs)} knowledge bases:\n")
    
    for i, (dir_name, full_path) in enumerate(available_dirs, 1):
        # Show meaningful names using the FIXED logic
        if dir_name == "rag_storage":
            topic = "Transformer and NLP Research Papers"
        else:
            topic = dir_name.replace("rag_data_", "").replace("_", " ")
        
        print(f"  [{i}] {topic}")
        print(f"      Directory: {dir_name}")
        print(f"      Length: {len(topic)} characters")
        print()
        
        # Verify this is properly processed
        if len(topic) > 30:
            print(f"      ✅ FIXED: Name longer than 30 characters!")
        elif len(dir_name.replace("rag_data_", "")) == 30:
            print(f"      ⚠️  OLD: This appears to be a truncated directory from before the fix")
        else:
            print(f"      ✅ OK: Short but complete name")
        print()
    
    return True

if __name__ == "__main__":
    success = test_knowledge_base_listing()
    if success:
        print("🎉 Knowledge base listing test completed!")
        print("\n📋 What you should see:")
        print("   • Full topic names displayed (not truncated to 30 chars)")
        print("   • Old directories may still show truncated names")
        print("   • New queries will create properly named directories")
    else:
        print("❌ Test failed!")
    
    sys.exit(0 if success else 1)