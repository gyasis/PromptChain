# Product Requirements Document (PRD)
## Sci-Hub MCP Server

### **1. Executive Summary**

The Sci-Hub MCP Server is a Model Context Protocol (MCP) implementation that provides programmatic access to academic papers through Sci-Hub's repository. This server enables AI assistants and applications to search, retrieve metadata, and download academic papers using various search methods including DOI, title, and keyword-based queries.

### **2. Product Overview**

#### **2.1 Purpose**
- Provide seamless access to academic papers through Sci-Hub
- Enable AI assistants to retrieve and cite academic sources
- Support research workflows with automated paper discovery
- Bridge the gap between AI tools and academic literature

#### **2.2 Target Users**
- AI assistants and chatbots
- Research automation tools
- Academic writing assistants
- Literature review automation systems
- Educational platforms

### **3. Technical Architecture**

#### **3.1 Technology Stack**
- **Framework**: FastMCP 2.0
- **Language**: Python 3.10+
- **Dependencies**: 
  - `fastmcp>=2.0.0`
  - `requests>=2.31.0`
  - `beautifulsoup4>=4.12.0`
  - `scihub`
- **Package Manager**: uv
- **Transport**: STDIO (default), HTTP, SSE

#### **3.2 Core Components**
- **MCP Server**: FastMCP implementation with 5 tools
- **Search Engine**: CrossRef API integration for metadata
- **PDF Access**: Sci-Hub integration for paper retrieval
- **Error Handling**: Comprehensive timeout and error management

### **4. Available Tools**

#### **4.1 `search_scihub_by_doi`**
**Purpose**: Retrieve academic papers using Digital Object Identifiers (DOIs)

**Parameters**:
- `doi` (str): Unique identifier for academic papers (e.g., "10.1038/nature09492")

**Returns**:
```json
{
  "doi": "10.1038/nature09492",
  "title": "Paper Title",
  "author": "Author Name",
  "year": "2023",
  "pdf_url": "https://sci-hub.se/...",
  "status": "success"
}
```

**Use Cases**:
- Direct paper retrieval when DOI is known
- Citation verification
- Academic reference lookup

**Performance**: 30-90 seconds (Sci-Hub API dependent)

#### **4.2 `search_scihub_by_title`**
**Purpose**: Find papers using their exact or partial title

**Parameters**:
- `title` (str): Paper title or partial title

**Returns**:
```json
{
  "doi": "10.1038/nature09492",
  "title": "Paper Title",
  "author": "Author Name", 
  "year": "2023",
  "pdf_url": "https://sci-hub.se/...",
  "status": "success"
}
```

**Use Cases**:
- Finding papers when title is known
- Literature review automation
- Research paper discovery

**Performance**: 45-90 seconds (CrossRef + Sci-Hub APIs)

#### **4.3 `search_scihub_by_keyword`**
**Purpose**: Discover papers using keywords, phrases, or research topics

**Parameters**:
- `keyword` (str): Search terms, phrases, or research topics
- `num_results` (int, optional): Number of results (default: 10, max: 100)

**Returns**:
```json
[
  {
    "title": "Paper Title",
    "author": "Author Name",
    "year": "2023",
    "doi": "10.1038/nature09492",
    "status": "metadata_only",
    "note": "Use search_scihub_by_doi to get PDF URL"
  }
]
```

**Search Capabilities**:
- Single keywords: `"machine learning"`
- Multiple keywords: `"artificial intelligence medicine 2023"`
- Phrases: `"deep learning neural networks"`
- Subjects: `"climate change adaptation"`

**Use Cases**:
- Research topic exploration
- Literature review generation
- Academic paper discovery
- Research trend analysis

**Performance**: 10-30 seconds (metadata only)

#### **4.4 `download_scihub_pdf`**
**Purpose**: Download PDF files of academic papers

**Parameters**:
- `pdf_url` (str): Direct PDF URL from search results
- `output_path` (str): Local file path for saving PDF

**Returns**:
```json
"PDF successfully downloaded to /path/to/paper.pdf"
```

**Use Cases**:
- Local paper storage
- Offline research
- Document processing workflows

**Performance**: Variable (depends on file size and network)

#### **4.5 `get_paper_metadata`**
**Purpose**: Retrieve comprehensive metadata for papers using DOI

**Parameters**:
- `doi` (str): Digital Object Identifier

**Returns**:
```json
{
  "doi": "10.1038/nature09492",
  "title": "Paper Title",
  "author": "Author Name",
  "year": "2023",
  "pdf_url": "https://sci-hub.se/...",
  "status": "success"
}
```

**Use Cases**:
- Citation generation
- Bibliography creation
- Academic reference management

**Performance**: 30-90 seconds

### **5. Integration Guide**

#### **5.1 Installation**
```bash
# Clone the repository
git clone <repository-url>
cd Sci-Hub-MCP-Server

# Install dependencies
uv sync

# Run the server
uv run fastmcp dev sci_hub_server.py
```

#### **5.2 MCP Configuration**
Add to your MCP client configuration:
```json
{
  "scihub": {
    "command": "uv",
    "args": ["run", "fastmcp", "dev", "sci_hub_server.py"],
    "cwd": "/path/to/Sci-Hub-MCP-Server"
  }
}
```

#### **5.3 API Usage Examples**

**DOI Search**:
```python
result = await client.call_tool("search_scihub_by_doi", {
  "doi": "10.1038/nature09492"
})
```

**Keyword Search**:
```python
papers = await client.call_tool("search_scihub_by_keyword", {
  "keyword": "machine learning healthcare",
  "num_results": 5
})
```

**PDF Download**:
```python
result = await client.call_tool("download_scihub_pdf", {
  "pdf_url": "https://sci-hub.se/...",
  "output_path": "./papers/paper.pdf"
})
```

### **6. Performance Characteristics**

#### **6.1 Timeout Settings**
- **Server Timeout**: 120 seconds
- **Sci-Hub API**: 90 seconds
- **HTTP Requests**: 90 seconds
- **Tool Execution**: 90 seconds maximum

#### **6.2 Response Times**
- **Keyword Search**: 10-30 seconds
- **DOI Search**: 30-90 seconds
- **Title Search**: 45-90 seconds
- **PDF Download**: Variable

#### **6.3 Rate Limits**
- **CrossRef API**: 50 requests/second
- **Sci-Hub**: No official limits (use responsibly)

### **7. Error Handling**

#### **7.1 Common Error Scenarios**
- **Timeout Errors**: Clear messages for slow responses
- **Network Issues**: Graceful degradation
- **Paper Not Found**: Informative error messages
- **Invalid DOI**: Validation and helpful suggestions

#### **7.2 Error Response Format**
```json
{
  "error": "Request timed out after 90 seconds. Sci-Hub may be slow or unavailable.",
  "status": "error"
}
```

### **8. Use Case Scenarios**

#### **8.1 AI Research Assistant**
```python
# 1. Search for recent papers on a topic
papers = await search_scihub_by_keyword("quantum computing 2024", 10)

# 2. Get detailed info for specific paper
paper = await search_scihub_by_doi("10.1038/nature12345")

# 3. Download for analysis
await download_scihub_pdf(paper["pdf_url"], "./research/paper.pdf")
```

#### **8.2 Literature Review Automation**
```python
# 1. Find papers by topic
papers = await search_scihub_by_keyword("machine learning healthcare", 20)

# 2. Get metadata for each
for paper in papers:
    if paper["doi"]:
        metadata = await get_paper_metadata(paper["doi"])
        # Process metadata for review
```

#### **8.3 Academic Writing Assistant**
```python
# 1. Find papers by title
paper = await search_scihub_by_title("Deep Learning for Medical Imaging")

# 2. Generate citation
citation = f"{paper['author']} ({paper['year']}). {paper['title']}. DOI: {paper['doi']}"
```

### **9. Security and Compliance**

#### **9.1 Data Privacy**
- No user data collection
- No persistent storage of search queries
- Local PDF storage only

#### **9.2 Responsible Usage**
- Respect academic access policies
- Use for legitimate research purposes
- Implement appropriate rate limiting

### **10. Limitations and Considerations**

#### **10.1 Current Limitations**
- No boolean search operators
- No field-specific search (title vs abstract)
- No exact phrase matching
- Dependent on Sci-Hub availability

#### **10.2 Future Enhancements**
- Advanced search filters
- Boolean operators support
- Field-specific search
- Citation format support
- Integration with other academic databases

### **11. Support and Maintenance**

#### **11.1 Dependencies**
- Sci-Hub service availability
- CrossRef API access
- Network connectivity

#### **11.2 Monitoring**
- Response time tracking
- Error rate monitoring
- API availability checks

### **12. Conclusion**

The Sci-Hub MCP Server provides a robust, scalable solution for integrating academic paper access into AI applications and research workflows. With comprehensive error handling, flexible search capabilities, and clear documentation, it enables developers to build sophisticated academic research tools while maintaining responsible usage practices.

The server's modular design and MCP compliance ensure easy integration with existing AI platforms and research automation systems, making academic literature more accessible to AI-powered applications. 