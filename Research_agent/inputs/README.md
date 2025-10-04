# Document Input Directory

Place research documents here for automatic processing:

## Supported Formats:
- PDF files (.pdf) - Research papers, reports
- Text files (.txt) - Preprocessed content
- Any text content for RAG processing

## Subdirectories:
- pdf/ - Place PDF documents here
- txt/ - Place text documents here  
- arxiv/ - ArXiv papers
- pubmed/ - PubMed papers
- manual/ - Manually curated documents

## Processing:
Documents placed here will be automatically:
1. Parsed and indexed by LightRAG
2. Processed by PaperQA2 for Q&A extraction
3. Converted to knowledge graphs by GraphRAG
4. Made available for research queries

The system monitors this directory and processes new files automatically.
