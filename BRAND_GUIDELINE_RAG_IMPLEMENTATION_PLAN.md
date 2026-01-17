# Brand Guideline RAG System - Implementation Plan
## Using ColPali Vision-RAG + FastMCP + Gemini Vision Platform Code Generation

**CONFIRMED APPROACH: ColPali Vision-RAG Setup**
- This system uses **ColPali** for vision-based document retrieval (as described in "The Future of RAG will be with Vision" article)
- PDFs are converted to images and indexed with ColPali model
- Retrieval returns actual page images preserving exact visual fidelity
- Gemini Vision extracts design specs and generates platform-specific code (React, Flutter, SwiftUI, Vue, Tailwind, etc.)
- **Not limited to CSS** - supports any platform/library as long as design specs are extracted correctly

---

## PHASE 1: Foundation & Environment Setup

### 1.1 Infrastructure Preparation
- Set up development environment with Python 3.9+
- Install `uv` package manager for dependency management
- **Provision hardware for ColPali (LOCAL MODEL):**
  - **GPU recommended** (NVIDIA CUDA, Apple MPS, or CPU fallback)
  - ColPali runs locally on your hardware (not a hosted API)
  - ~10GB storage for model download (one-time)
  - GPU memory: ~8-16GB recommended for efficient inference
  - CPU works but 10x+ slower (not recommended for production)
- Set up version control (Git) and project structure

### 1.2 Core Library Installation (using uv)
- Install PyTorch (with CUDA support if GPU available, or CPU version)
- Install Transformers library (Hugging Face) for ColPali model access
- Install PDF processing libraries: `pdf2image`, `pypdf` (or `PyMuPDF`)
- Install image processing: `Pillow` (PIL)
- Install ColPali dependencies: `torch`, `transformers`, `tqdm`
- Install FastMCP framework: `fastmcp` (Python package)
- Install Gemini API client: `google-generativeai`
- Install vector database (optional for hybrid approach): `qdrant-client` or `chromadb`

### 1.3 API Keys & Configuration
- Obtain Google Gemini API key for Gemini Vision
- Set up environment variables (.env file) for API keys
- Configure authentication for all external services

---

## PHASE 2: PDF Processing & Visual Fidelity Preservation

### 2.1 Brand Guideline PDF Analysis
- Audit all brand guideline PDFs (versions, formats, structure)
- Identify visual elements that must be preserved (fonts, colors, spacing, typography)
- Document document structure (headings, sections, design examples)

### 2.2 PDF to Image Conversion Pipeline
- Set up high-resolution PDF to image conversion (300+ DPI recommended)
- Convert each PDF page to PNG/JPEG images (preserve exact visual appearance)
- Store original page images in organized directory structure
- Ensure lossless image formats to maintain visual fidelity

### 2.3 Text Extraction (Parallel Processing)
- Extract text from PDF pages using `pypdf` or `PyMuPDF`
- Store extracted text alongside corresponding page images
- Maintain page number mapping between images and text

### 2.4 Metadata Collection
- Create metadata for each page: page number, file name, section headings
- Tag design elements: fonts, colors, spacing specifications
- Build metadata structure for later retrieval and filtering

---

## PHASE 3: ColPali Vision-RAG Setup & Indexing

### 3.1 ColPali Model Setup (LOCAL MODEL - Can Be Hosted on Cloud)

**IMPORTANT: For MCP Server Usage - Hugging Face Inference Endpoints with Scale-to-Zero**
**Important: ColPali runs on your infrastructure, but can be deployed to cloud platforms**

**Option A: Local Deployment (Standard)**
- Download ColPali model from Hugging Face (`vidore/colpali` or `vidore/colpali-v1.3-hf`)
- Load base model: `google/paligemma-3b-mix-448` (downloaded automatically)
- Load ColPali adapter weights (~10GB download, one-time, stored locally)
- **Hardware Requirements:**
  - GPU recommended (CUDA for NVIDIA, MPS for Mac, or CPU fallback)
  - Model runs entirely on your hardware (no API calls)
  - GPU significantly faster than CPU (10x+ speed difference)
- Configure device mapping: `device_map="auto"` (auto-detects GPU/CPU)
- Set up model processor: `AutoProcessor.from_pretrained(model_name)`

**Option B: Cloud Hosting (Recommended for Production)**
- **Hugging Face Inference Endpoints** (Easiest - Managed Service):
  - Deploy ColPali to Hugging Face Inference Endpoints
  - Supports AWS, Azure, GCP
  - Auto-scaling, managed infrastructure
  - Pay-per-minute billing
  - GPU options: Tesla T4, A100, L4, L40S
  - Requires Hugging Face subscription + linked credit card
  
- **Azure Deployment:**
  - **Azure Machine Learning**: Import model, create custom environment, deploy to online endpoint
  - **Azure Functions**: Containerize with Docker, deploy container
  - **Azure Kubernetes Service (AKS)**: Containerize and deploy to AKS
  - GPU instances available (NC-series, ND-series)
  
- **AWS Deployment:**
  - **Amazon SageMaker**: Deploy model endpoint with GPU instances
  - **EC2 GPU Instances**: Deploy on EC2 with GPU (g4dn, p3, p4d instances)
  - **ECS/EKS**: Containerize and deploy to container services
  - Can integrate with Amazon Nova for AI pipeline orchestration
  
- **Google Cloud Platform:**
  - **Vertex AI**: Deploy model to Vertex AI endpoints
  - **Cloud Run**: Containerize and deploy (with GPU support)
  - **GKE (Google Kubernetes Engine)**: Deploy containerized model
  - GPU instances: T4, A100, L4
  
- **BentoML Deployment** (API Framework):
  - Deploy ColPali as inference API using BentoML
  - Supports adaptive batching for multi-vector embeddings
  - Can deploy to any cloud platform
  - Optimizes memory usage for ColPali's multi-vector approach

**Recommendation for MCP Server Usage:**
- **Use Hugging Face Inference Endpoints with Scale-to-Zero enabled**
- **Why:** Perfect for intermittent MCP tool calls
  - MCP server makes tool call → ColPali endpoint wakes up → Processes → Scales to zero
  - $0 cost when idle (between tool calls)
  - Only pay for actual inference time
  - Cold start (2-5 min) is acceptable for MCP usage patterns
- **Configuration:**
  - Enable "Scale to Zero" when creating endpoint
  - Set idle timeout based on your usage (15 min - 1 hour typical)
  - Monitor request patterns to optimize timeout

### 3.2 Document Indexing Pipeline
- Process each page image through ColPali model
- Generate visual embeddings for each page (ColPali creates multi-vector embeddings)
- Store embeddings with corresponding page images and metadata
- Build index structure that maps embeddings to page images

### 3.3 Index Storage & Management
- Decide on storage approach: in-memory (for small datasets) or persistent storage
- Create data structure linking: embeddings → page images → text → metadata
- Implement index persistence (save/load functionality)

---

## PHASE 4: FastMCP Server Development

### 4.1 FastMCP Server Initialization
- Create FastMCP server instance using `uv` environment
- Configure server name and basic settings
- Set up server to run with stdio transport (for MCP protocol)

### 4.2 MCP Tools Development
- **Tool 1: `search_brand_guidelines`** - Query the ColPali index for relevant pages
  - Input: user query (text)
  - Process: Convert query to ColPali embedding, search index
  - Output: Top K relevant page images + text + metadata
  
- **Tool 2: `get_design_specification`** - Retrieve specific design element
  - Input: design element type (typography, colors, spacing)
  - Process: Search with ColPali, filter by metadata tags
  - Output: Relevant page images showing that design element

- **Tool 3: `get_page_image`** - Retrieve exact page image by page number
  - Input: page number, document name
  - Process: Direct lookup in stored images
  - Output: Base64 encoded page image

### 4.3 MCP Resources Setup
- **Resource 1: `brand://guidelines/{section}`** - Access specific guideline sections
- **Resource 2: `brand://images/{page_number}`** - Access page images
- **Resource 3: `brand://metadata`** - Access document metadata

### 4.4 Server Testing
- Test FastMCP server startup and shutdown
- Verify tools are properly registered and callable
- Test resource access patterns

---

## PHASE 5: Gemini Vision Integration for Design Specs Extraction & Platform Code Generation
**Based on gemini-mcp `interpret_image` tool pattern**

### 5.1 Gemini Vision API Setup (Following gemini-mcp Pattern)
- Configure Google Generative AI client using `google.genai.Client` (unified SDK)
- Set up Gemini Vision model: `gemini-2.0-flash-001` (or `gemini-pro-vision`)
- Implement authentication using API key from environment variables
- Set up error handling similar to gemini-mcp's approach

### 5.2 Image Input Handling (Following gemini-mcp Pattern)
- Support multiple image input formats (like `interpret_image` tool):
  - **Local file paths**: Direct path to page images from ColPali retrieval
  - **Base64 encoded**: Convert page images to base64 for inline processing
  - **URLs**: If images are served via web (optional enhancement)
- Implement image format detection (PNG, JPEG, etc.)
- Handle large images (>20MB) using Gemini File API with upload/polling
- Support multiple images per request (up to 3,600 like gemini-mcp)

### 5.3 Design Specs Extraction & Platform Code Generation Prompt Design (Key Integration Point)
- Design prompt template that combines:
  - **Image**: Retrieved brand guideline page image(s) from ColPali
  - **User Question**: "What is the typography and spacing?" or "What are the colors?"
  - **Target Platform**: User specifies platform/library (React, Flutter, SwiftUI, Vue, Tailwind CSS, etc.)
  - **Context**: Extracted text from that page (optional, for additional context)
  - **Instruction**: "Analyze this brand guideline page image. Extract the design specifications including typography (font family, sizes, weights, line heights), spacing (margins, padding, gaps), colors (hex codes, RGB values), layout (grid, flexbox patterns), and any other design tokens. Generate code in {target_platform} format that implements these specifications visually. The code should accurately represent the visual design shown in the image."
- Create specialized prompts for different query types:
  - Typography queries: "Extract font family, sizes, weights, line heights. Generate {platform} code for typography"
  - Spacing queries: "Extract margin and padding values. Generate {platform} code with proper spacing"
  - Color queries: "Extract color palette with hex codes. Generate {platform} code with color tokens"
  - Layout queries: "Extract layout specifications. Generate {platform} code for layout (grid/flexbox/stack/etc)"
  - Complete component: "Extract all design specs. Generate a complete {platform} component matching the visual design"
- Support multiple platforms/libraries:
  - **Web**: React, Vue, Angular, Svelte, HTML/CSS, Tailwind CSS, styled-components
  - **Mobile**: Flutter/Dart, React Native, SwiftUI, Kotlin Compose
  - **Design Systems**: Material-UI, Chakra UI, Ant Design, Mantine
  - **CSS Frameworks**: Tailwind, Bootstrap, Bulma
  - **Platform-specific**: Next.js, Remix, Nuxt

### 5.4 Gemini Vision API Call Pattern (Following gemini-mcp)
- Use `client.models.generate_content()` with:
  - Model: `gemini-2.0-flash-001`
  - Contents: Array with image(s) + prompt text
  - Config: Temperature (0.3-0.5 for accuracy), max_output_tokens (8192)
- Handle image data:
  - Small images (<20MB): Send inline as `Part.from_bytes()`
  - Large images (>20MB): Upload via File API, poll for ACTIVE state, use `Part.from_uri()`
- Support multiple images: Send all retrieved page images in one request for comprehensive analysis

### 5.5 Platform Code Extraction & Validation
- Parse Gemini Vision response to extract:
  - Platform-specific code blocks (React, Flutter, SwiftUI, etc. - detected by language tag)
  - Design tokens (colors, fonts, spacing values) as structured data
  - Explanations and context about the design choices
- Validate code syntax (optional: use platform-specific validators)
- Structure output:
  - Generated platform code (React component, Flutter widget, SwiftUI view, etc.)
  - Design tokens (JSON format): `{colors: [...], fonts: [...], spacing: [...], layout: [...]}`
  - Platform/library used
  - Source page reference (page number, section name)
  - Confidence/explanation from Gemini
  - Visual representation instructions (if applicable)

---

## PHASE 6: Integration & Workflow Assembly

### 6.1 End-to-End Query Flow (Following gemini-mcp Pattern)
- **Step 1:** User submits query (e.g., "What is the recommended typography and spacing?")
- **Step 2:** FastMCP tool receives query
- **Step 3:** Query processed through ColPali search function
- **Step 4:** ColPali returns top K relevant page images + text + metadata
- **Step 5:** Convert page images to format compatible with Gemini API:
  - Read image files from ColPali results
  - Convert to base64 or prepare for File API (if large)
- **Step 6:** Build Gemini Vision prompt:
  - Combine user question with design specs extraction + platform code generation instruction
  - Include target platform/library (from user query or default)
  - Include extracted text as context (optional)
  - Format: "User question: {query}. Target platform: {platform}. Analyze this brand guideline page and extract design specifications (typography, spacing, colors, layout). Generate {platform} code that implements these specifications visually."
- **Step 7:** Call Gemini Vision API (like `interpret_image` tool):
  - Send image(s) + prompt to `client.models.generate_content()`
  - Use `gemini-2.0-flash-001` model
  - Receive response with platform-specific code and design tokens
- **Step 8:** Parse and structure response:
  - Extract platform code from response (detect language: jsx, dart, swift, vue, etc.)
  - Extract design tokens (colors, fonts, spacing, layout)
  - Identify platform/library used
  - Link back to source page images and metadata
- **Step 9:** Assemble final response:
  - Retrieved page images (for visual reference)
  - Extracted text (for context)
  - Generated platform code (React/Flutter/SwiftUI/etc.)
  - Design tokens (structured data)
  - Platform/library identifier
  - Source metadata (page numbers, sections)

### 6.2 FastMCP Tool: `generate_code_from_guidelines` (Following gemini-mcp Pattern)
- Combine ColPali retrieval + Gemini Vision design specs extraction + platform code generation
- **Input Parameters:**
  - `query`: User question (e.g., "What is the typography and spacing?")
  - `platform`: Target platform/library (e.g., "react", "flutter", "swiftui", "vue", "tailwind", "nextjs")
  - `design_element`: Optional filter (typography, colors, spacing, layout)
  - `num_pages`: Number of pages to retrieve (default: 3)
- **Process:**
  1. Call ColPali search with user query
  2. Retrieve top K page images + text + metadata
  3. Convert images to Gemini-compatible format (base64 or File API)
  4. Build prompt combining user question + platform target + design specs extraction instruction
  5. Call Gemini Vision API (following `interpret_image` pattern)
  6. Parse response to extract platform code and design tokens
- **Output Structure:**
  - `platform_code`: Generated code in target platform format (React component, Flutter widget, etc.)
  - `platform`: Platform/library used (react, flutter, swiftui, etc.)
  - `design_tokens`: JSON with colors, fonts, spacing values, layout specs
  - `source_images`: Base64 encoded page images (for reference)
  - `source_text`: Extracted text from pages
  - `metadata`: Page numbers, section names, file paths
  - `explanation`: Gemini's explanation of the design specs and code

### 6.3 Response Formatting
- Structure response to include:
  - Retrieved page images (base64 or URLs)
  - Extracted text from those pages
  - Generated CSS code
  - Design tokens (colors, fonts, spacing values)
  - Metadata (page numbers, section names)

---

## PHASE 7: Advanced Features & Optimization

### 7.1 Hybrid Search Enhancement (Optional)
- Add text-based search alongside ColPali vision search
- Implement Reciprocal Rank Fusion (RRF) to combine results
- Use vector database (Qdrant/ChromaDB) for text embeddings if needed

### 7.2 Caching Strategy
- Cache ColPali embeddings (don't re-index on every query)
- Cache Gemini Vision responses for common queries
- Implement cache invalidation strategy

### 7.3 Performance Optimization
- Batch processing for multiple pages
- Async processing for Gemini API calls
- Optimize image sizes for faster processing

### 7.4 Error Handling & Fallbacks
- Handle ColPali model loading failures
- Handle Gemini API rate limits and errors
- Implement fallback to text-only search if vision fails

---

## PHASE 8: Testing & Validation

### 8.1 Visual Fidelity Testing
- Compare retrieved page images with original PDF pages
- Verify fonts, colors, spacing are preserved exactly
- Test with various brand guideline PDF formats

### 8.2 Retrieval Accuracy Testing
- Test queries: typography, colors, spacing, layout
- Verify ColPali returns relevant pages
- Measure retrieval precision and recall

### 8.3 Platform Code Generation Quality Testing
- Test Gemini Vision code generation accuracy across multiple platforms
- Test with different platforms: React, Flutter, SwiftUI, Vue, Tailwind
- Compare generated code with actual design specifications
- Validate extracted design tokens (colors, fonts, spacing, layout)
- Verify code compiles/runs in target platform (if possible)
- Test platform-specific code patterns (React hooks, Flutter widgets, SwiftUI views)

### 8.4 End-to-End Workflow Testing
- Test complete flow: query → retrieval → CSS generation
- Verify FastMCP tools work correctly
- Test error scenarios and edge cases

---

## PHASE 9: Deployment & Production

### 9.1 Production Environment Setup

**ColPali Deployment Options:**

**Option 1: Hugging Face Inference Endpoints (Recommended - Easiest)**
- Deploy ColPali to Hugging Face Inference Endpoints
- Managed infrastructure (no Kubernetes/CUDA management)
- **Auto-scaling with Scale-to-Zero (Perfect for MCP Servers):**
  - **Scale to Zero:** Endpoint scales down to zero replicas when idle
  - **Idle Timeout:** Configurable (typically 15 minutes to 1 hour)
  - **Cost:** $0 when scaled to zero (no charges during idle periods)
  - **MCP Server Pattern:** 
    - Tool call → Endpoint wakes up → Processes request → Scales to zero after idle timeout
    - Perfect for intermittent usage (MCP tool calls)
    - Only pay for actual inference time
  - **Cold Start:** 2-5 minutes when waking from zero (model loading time)
  - **Scaling Frequency:** Scales up every 1 minute, scales down every 2 minutes
- Supports AWS, Azure, GCP
- GPU options: T4, A100, L4, L40S
- Pay-per-minute billing (only when active)
- **Pros:** 
  - Easiest setup, managed infrastructure
  - Scale-to-zero = $0 cost when idle (perfect for MCP servers)
  - Auto-scaling handles traffic spikes
- **Cons:** 
  - Requires Hugging Face subscription
  - Cold start latency (2-5 min) when waking from zero
  - Less control than self-hosted

**Option 2: Cloud Provider ML Services**
- **Azure:** Azure Machine Learning, Azure Functions, AKS
- **AWS:** SageMaker, EC2 GPU instances, ECS/EKS
- **GCP:** Vertex AI, Cloud Run, GKE
- **Pros:** Full control, integrates with existing cloud infrastructure
- **Cons:** More setup required, manage infrastructure yourself

**Option 3: Self-Hosted Cloud GPU Instance**
- Deploy on cloud GPU instance (AWS EC2 GPU, Azure NC-series, GCP GPU VMs)
- Download and load model on your server
- Use containerization (Docker) for consistent deployment
- **Pros:** Full control, can be cheaper for high usage
- **Cons:** More management overhead

**FastMCP Server Deployment:**
- Deploy FastMCP server (can be on same infrastructure or separate)
- Set up proper process management (systemd, supervisor, etc.)
- Configure API endpoints and authentication

**Monitoring & Logging:**
- Set up monitoring for ColPali inference performance
- Monitor GPU utilization and costs
- Track query performance and accuracy
- Set up alerts for failures

**Cost Considerations:**

**Hugging Face Inference Endpoints (Best for MCP Servers):**
- **Scale-to-Zero Enabled:** $0 cost when idle (no requests)
- **Pay-per-minute:** Only charged when endpoint is active (processing requests)
- **MCP Server Usage Pattern:**
  - Tool call → Endpoint wakes (2-5 min cold start) → Processes → Scales to zero after idle timeout
  - If tool calls are >15 minutes apart: Endpoint scales to zero between calls = $0 cost
  - If tool calls are frequent (<15 min apart): Endpoint stays active = pay for runtime
- **Cost Optimization:**
  - Configure idle timeout based on your MCP server usage pattern
  - Monitor request frequency to optimize timeout settings
  - For intermittent usage (typical MCP pattern): Scale-to-zero = significant cost savings
- **Cold Start Trade-off:** 2-5 minute delay when waking from zero (acceptable for MCP tool calls)

**Cloud ML Services (Azure ML, SageMaker, Vertex AI):**
- Pay for GPU instance time (even when idle)
- No scale-to-zero (or limited)
- More expensive for intermittent usage

**Self-Hosted:**
- Pay for GPU instance continuously
- Full control but no automatic cost optimization

**Recommendation for MCP Servers:**
- **Use Hugging Face Inference Endpoints with Scale-to-Zero enabled**
- Perfect for intermittent MCP tool calls
- $0 cost when idle, only pay during actual inference
- Cold start acceptable for MCP server usage patterns (2-5 min delay is fine)

### 9.2 API Endpoint Configuration
- Configure FastMCP server endpoints
- Set up authentication/authorization if needed
- Configure rate limiting

### 9.3 Monitoring & Maintenance
- Set up monitoring for ColPali model performance
- Monitor Gemini API usage and costs
- Track query performance and accuracy
- Set up alerts for failures

---

## KEY ARTICLES & REFERENCE IMPLEMENTATIONS

1. **"The Future of RAG will be with Vision: End to End Example with ColPali"**
   - ColPali model setup and indexing process
   - Vision-based retrieval workflow
   - Integration with Vision Language Models for generation

2. **"Revolutionizing RAG by Integrating Vision Models"**
   - Dual-stream architecture concepts (can be adapted)
   - Qdrant multi-vector storage (optional enhancement)
   - Vision + text processing patterns

3. **gemini-mcp Server Implementation** (`/home/gyasis/Documents/code/gemini-mcp`)
   - **Key Pattern: `interpret_image` tool** - Shows how to send images to Gemini Vision
   - **Image Handling**: Supports file paths, URLs, base64 encoding
   - **API Pattern**: Uses `google.genai.Client` with `generate_content()` method
   - **Multi-Image Support**: Can process up to 3,600 images in one request
   - **Large File Handling**: Uses File API for files >20MB with polling
   - **Prompt Engineering**: Demonstrates how to structure prompts for image analysis
   - **This pattern is directly applied to Phase 5 and Phase 6** of this implementation

---

## REQUIRED LIBRARIES SUMMARY

**Core Vision-RAG:**
- `torch` (PyTorch)
- `transformers` (Hugging Face)
- `pdf2image`
- `pypdf` or `PyMuPDF`
- `Pillow`

**ColPali Specific (LOCAL MODEL - Can Be Cloud Hosted):**
- ColPali model: `vidore/colpali` or `vidore/colpali-v1.3-hf` from Hugging Face
- Base model: `google/paligemma-3b-mix-448` (auto-downloaded)
- **Deployment Options:**
  - **Local:** Runs on your hardware (GPU/CPU)
  - **Cloud Hosted:** Hugging Face Inference Endpoints, Azure ML, AWS SageMaker, GCP Vertex AI
  - **Containerized:** Docker deployment to any cloud platform
- Model download: ~10GB (one-time)
- Requires: GPU (recommended) or CPU (slower) for inference
- **Cloud Hosting Tools:**
  - `bentoml` (for API deployment)
  - Hugging Face Inference Endpoints (managed service)
  - Cloud provider ML services (Azure ML, SageMaker, Vertex AI)

**FastMCP Server:**
- `fastmcp` (Python package)
- `uv` (package manager)

**Gemini Vision (Following gemini-mcp Pattern):**
- `google-generativeai` (unified Google Gen AI SDK)
- Use `google.genai.Client` for API access
- Use `google.genai.types` for Part, GenerateContentConfig
- Pattern matches gemini-mcp's `interpret_image` tool implementation

**Optional Enhancements:**
- `qdrant-client` (for hybrid search)
- `sentence-transformers` (for text embeddings)
- `chromadb` (alternative vector DB)

---

## ARCHITECTURE CONFIRMATION

**YES - This uses ColPali Vision-RAG Setup:**
- ✅ PDFs converted to images (preserves visual fidelity)
- ✅ **ColPali model runs LOCALLY** (downloaded from Hugging Face, runs on your hardware)
- ✅ ColPali model indexes page images directly (local inference)
- ✅ Vision-based retrieval (no OCR preprocessing needed)
- ✅ Returns actual page images (exact visual appearance)
- ✅ Gemini Vision processes images to extract design specs and generate platform-specific code (following gemini-mcp `interpret_image` pattern)
- **Key Difference:** 
- ColPali = Can run locally OR be hosted on cloud (Hugging Face, Azure, AWS, GCP)
- Gemini Vision = Hosted API (Google's servers)
- **Recommendation:** Host ColPali on cloud (Hugging Face Inference Endpoints) for production, run locally for development

**Key Integration Pattern (from gemini-mcp):**
- ✅ Retrieve page images with ColPali
- ✅ Send images to Gemini Vision API using `interpret_image`-like pattern
- ✅ Gemini reads images and understands design questions
- ✅ Gemini extracts design specs (typography, spacing, colors, layout) 
- ✅ Gemini generates platform-specific code (React, Flutter, SwiftUI, Vue, Tailwind, etc.)
- ✅ Support multiple platforms/libraries (not limited to CSS)
- ✅ Support multiple image formats (file paths, base64, URLs)
- ✅ Handle large images via File API with polling

**NOT using:**
- ❌ LightRAG (graph-based, not vision-first)
- ❌ Traditional text-only RAG
- ❌ Dual-stream Qdrant approach (optional enhancement only)

---

## NEXT STEPS

1. Review this implementation plan
2. Set up Phase 1 (environment and libraries)
3. Begin Phase 2 (PDF processing)
4. Progress through phases sequentially
5. Test each phase before moving to next

