# PyWrap Framework - Product Requirements Document (PRD)

## Document Information
- **Document Title**: PyWrap Framework - Python CLI Wrapper Framework
- **Version**: 1.0.0
- **Date**: January 13, 2025
- **Author**: Development Team
- **Status**: Draft

## Executive Summary

PyWrap is a **living development framework** that transforms Python scripts into npm-installable CLI tools with **zero configuration for 90% of use cases**. The framework emphasizes Typer and Rich for Python CLI development, provides intelligent project analysis, and offers a **continuous development workflow** where the Node.js wrapper automatically stays in sync with your Python script as you develop.

**Key Insight**: Most CLI tools only need simple parameter passing from Node.js to Python - complex communication protocols are overkill for the majority of use cases.

## Problem Statement

### Current Pain Points
1. **Environment Management**: Users must manually manage Python environments and dependencies when using Python CLI tools
2. **Distribution Complexity**: Python tools require users to understand pip, virtual environments, and Python versioning
3. **CLI Development Overhead**: Building professional CLI tools with Python requires significant boilerplate setup
4. **Cross-Platform Issues**: Python CLI tools often have platform-specific installation and usage challenges
5. **Integration Barriers**: Node.js developers struggle to integrate Python tools into their workflows

### Market Opportunity
- **Target Users**: Python developers, Node.js developers, DevOps engineers, CLI tool creators
- **Market Size**: Growing demand for professional CLI tools and developer productivity tools
- **Competitive Advantage**: First comprehensive framework for Python-to-npm CLI wrapping with intelligent analysis

## Solution Overview

### Core Value Proposition
PyWrap transforms any Python script into a professional, npm-installable CLI tool while preserving the original Python logic and enhancing it with modern CLI capabilities.

### Key Differentiators
1. **Intelligent Analysis**: Automatically detects project structure, dependencies, and entry points
2. **Dual Mode Operation**: Can generate new projects OR wrap existing ones
3. **Typer + Rich Integration**: Built-in support for modern Python CLI development
4. **Living Development Workflow**: Wrapper automatically stays in sync with Python script changes
5. **80/20 Rule**: Simple parameter passing for 90% of use cases, advanced features when needed
6. **Zero Configuration**: Works out-of-the-box with intelligent defaults

## Product Requirements

### Functional Requirements

#### 1. Project Generation Mode (`pywrap init`)
- **Command**: `pywrap init <project-name> [--template <template>] [options]`
- **Templates Available**:
  - `cli`: Basic CLI tool with Typer and Rich
  - `api`: FastAPI backend with CLI wrapper
  - `library`: Python library with CLI interface
  - `web`: Web application with CLI management
  - `data`: Data processing tool with CLI
- **Auto-generation**:
  - Complete project structure
  - npm package configuration
  - Python environment setup (uv)
  - Documentation and examples
  - Testing framework setup

#### 2. Project Wrapping Mode (`pywrap wrap`)
- **Command**: `pywrap wrap <project-path> [options]`
- **Intelligent Analysis**:
  - Auto-detect CLI entry points
  - Identify dependencies and requirements
  - Parse existing CLI arguments and help text
  - Detect async vs sync patterns
  - Analyze project structure and type
- **Wrapper Generation**:
  - npm package with proper bin configuration
  - Node.js wrapper with communication layer
  - Preserve original Python structure
  - Generate appropriate documentation

#### 3. Template System (`pywrap scaffold`)
- **Command**: `pywrap scaffold <template> [project-path]`
- **Custom Templates**: Allow users to create and share custom templates
- **Template Repository**: Centralized template sharing and discovery
- **Template Validation**: Ensure templates follow best practices

#### 4. Communication Layer
- **Simple Parameter Passing (Default)**: Basic STDIO communication for 90% of use cases
- **Advanced Communication (Optional)**: Async, streaming, and interactive features when needed
- **Protocol Selection**: Auto-select optimal communication method based on project complexity
- **Error Handling**: Graceful error propagation and recovery
- **Performance Optimization**: Efficient data exchange with minimal overhead

#### 5. Development Workflow
- **Watch Mode**: Automatically detect Python script changes and update wrapper
- **Incremental Updates**: Sync specific components (parameters, commands, help text)
- **Smart Wrappers**: Auto-update capabilities built into generated wrappers
- **Hot Reloading**: Fast development cycle without manual intervention
- **Development vs Production**: Different modes for development and production use

### Non-Functional Requirements

#### 1. Performance
- **Startup Time**: < 500ms for basic commands
- **Communication Latency**: < 50ms for simple parameter passing
- **Memory Usage**: < 25MB baseline for simple wrappers
- **Scalability**: Support for projects with 100+ dependencies
- **Development Speed**: < 2 seconds for wrapper regeneration

#### 2. Reliability
- **Error Recovery**: 99% success rate for wrapper generation
- **Process Management**: Automatic cleanup of orphaned processes
- **Dependency Resolution**: Intelligent conflict resolution
- **Fallback Mechanisms**: Graceful degradation when features unavailable

#### 3. Usability
- **Learning Curve**: < 2 minutes to first working tool
- **Documentation**: Comprehensive examples and tutorials
- **Error Messages**: Clear, actionable error messages
- **Interactive Mode**: Guided setup for complex configurations
- **Development Workflow**: Seamless iteration without manual wrapper updates

#### 4. Compatibility
- **Python Versions**: 3.8+ support
- **Node.js Versions**: 16+ support
- **Operating Systems**: Windows, macOS, Linux
- **Package Managers**: uv, pip, poetry, conda support

## Technical Architecture

### Simplified Architecture (90% of Use Cases)

```
┌─────────────────────────────────────────────────────────────┐
│                    PyWrap Framework                        │
├─────────────────────────────────────────────────────────────┤
│  Core Engine (Python)           │  CLI Interface (Node.js) │
│  ├─ Project Analyzer            │  ├─ Command Parser       │
│  ├─ Template Engine             │  ├─ Project Generator    │
│  ├─ Wrapper Generator           │  ├─ Wrapper Manager      │
│  ├─ Development Workflow        │  └─ Package Manager      │
│  └─ Communication Manager       │                          │
├─────────────────────────────────────────────────────────────┤
│  Communication Layer (STDIO Default, Advanced Optional)    │
├─────────────────────────────────────────────────────────────┤
│  Generated Projects (npm + Python)                         │
│  ├─ package.json               │  ├─ main.py               │
│  ├─ bin/cli.js                 │  ├─ requirements.txt      │
│  └─ src/wrapper.js             │  └─ pyproject.toml        │
└─────────────────────────────────────────────────────────────┘
```

### Development Workflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Development Mode                         │
├─────────────────────────────────────────────────────────────┤
│  File Watcher (Chokidar)        │  Smart Wrapper            │
│  ├─ Monitor Python Script       │  ├─ Auto-Detect Changes  │
│  ├─ Trigger Updates             │  ├─ Incremental Sync     │
│  └─ Live Reload                 │  └─ Hot Restart          │
├─────────────────────────────────────────────────────────────┤
│  Update Manager (Incremental)                               │
│  ├─ Sync Parameters             │  ├─ Sync Commands         │
│  ├─ Sync Help Text             │  └─ Full Sync             │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Project Analyzer
- **Entry Point Detection**: Multiple strategies for finding CLI entry points
- **Dependency Analysis**: Parse requirements.txt, pyproject.toml, setup.py
- **Structure Mapping**: Understand project organization and patterns
- **Type Classification**: Identify project type (CLI, API, library, etc.)

#### 2. Template Engine
- **Template System**: YAML/TOML-based template definitions
- **Code Generation**: Jinja2-based code generation
- **Validation**: Template syntax and structure validation
- **Customization**: User-defined template variables and options

#### 3. Wrapper Generator
- **npm Package Generation**: Create package.json with proper configuration
- **Node.js Wrapper**: Generate communication and execution logic
- **Documentation**: Auto-generate README, help text, and examples
- **Testing**: Generate test framework and example tests

#### 4. Communication Manager
- **Protocol Selection**: Choose optimal communication method
- **Process Management**: Handle Python process lifecycle
- **Data Serialization**: Efficient data exchange formats
- **Error Handling**: Robust error propagation and recovery

### Communication Protocols

#### 1. STDIO (Default - 90% of Use Cases)
- **Pros**: Simple, reliable, works everywhere, zero configuration
- **Cons**: Limited for complex communication
- **Use Case**: Basic CLI tools, simple data exchange, parameter passing
- **Example**: File processing, data analysis, simple workflows

#### 2. Enhanced STDIO (10% of Use Cases)
- **Pros**: Simple with progress bars, interactive prompts
- **Cons**: Still limited for complex scenarios
- **Use Case**: Tools with progress updates, user input
- **Example**: Data processing with progress, interactive analysis

#### 3. ZeroMQ (Advanced - 5% of Use Cases)
- **Pros**: High performance, async, bidirectional
- **Cons**: Additional dependency, complexity
- **Use Case**: High-performance tools, complex communication
- **Example**: Real-time data processing, streaming applications

#### 4. gRPC (Enterprise - 5% of Use Cases)
- **Pros**: Type-safe, efficient, bidirectional
- **Cons**: Protocol buffer dependency, setup complexity
- **Use Case**: Enterprise tools, microservice communication
- **Example**: Large-scale data processing, enterprise integrations

### When to Use Advanced Communication

**Use Simple STDIO When:**
- Passing basic parameters (strings, numbers, booleans)
- Simple file processing or data analysis
- Basic CLI tools with standard output
- 90% of typical use cases

**Use Enhanced STDIO When:**
- Need progress bars or status updates
- Require user input during execution
- Want rich terminal output
- Still want simple setup

**Use Advanced Protocols When:**
- Real-time streaming data
- Complex bidirectional communication
- High-performance requirements
- Enterprise integration needs

## User Experience

### User Journey

#### 1. New User (Project Generation)
```
1. Install PyWrap: npm install -g pywrap
2. Generate project: pywrap init my-tool --template=cli
3. Navigate to project: cd my-tool
4. Install dependencies: npm install
5. Run tool: my-tool --help
6. Develop Python logic in python/main.py
7. Test and iterate
8. Publish: npm publish
```

#### 2. Development Workflow (Continuous Iteration)
```
1. Start development mode: pywrap dev my-tool --watch
2. Modify Python script (add parameters, commands, etc.)
3. Wrapper automatically updates (no manual intervention)
4. Test changes immediately: my-tool --help
5. Iterate rapidly without wrapper management overhead
6. Build for production when ready: pywrap build my-tool
```

#### 2. Existing User (Project Wrapping)
```
1. Navigate to existing project: cd ~/projects/my-script
2. Wrap project: pywrap wrap . --intelligent
3. Review generated wrapper
4. Install wrapper: npm install -g .
5. Use wrapped tool: my-script --help
6. Customize if needed
7. Distribute via npm
```

### Command Interface

```bash
# Main help
pywrap --help

# Project generation
pywrap init <name> [--template <template>] [--with-progress] [--with-interactive]

# Project wrapping
pywrap wrap <path> [--intelligent] [--preserve-structure] [--add-dependencies]

# Template management
pywrap scaffold <template> [path] [--force] [--customize]
pywrap scaffold --list
pywrap scaffold --info <template>

# Project management
pywrap build [path]
pywrap test [path]
pywrap publish [path]

# Configuration
pywrap config --set <key> <value>
pywrap config --get <key>
pywrap config --list
```

### Development Workflow Commands

#### Watch Mode (Continuous Development)
```bash
# Start development mode with auto-sync
pywrap dev reportbuilder --watch

# This automatically:
# - Monitors Python script changes
# - Updates Node.js wrapper
# - Restarts wrapper process
# - Shows live development logs
```

#### Incremental Updates
```bash
# Sync only parameters (fastest)
pywrap update reportbuilder --sync-parameters

# Sync only command structure
pywrap update reportbuilder --sync-commands

# Sync only help text and documentation
pywrap update reportbuilder --sync-help

# Full synchronization (slowest)
pywrap update reportbuilder --full-sync
```

## Implementation Plan

### Phase 1: Core Framework (Weeks 1-4)
- [ ] Project structure and architecture setup
- [ ] Basic CLI interface with Commander.js
- [ ] Project analyzer for entry point detection
- [ ] Simple template system for CLI projects
- [ ] Basic npm package generation
- [ ] Simple STDIO communication layer
- [ ] Basic wrapper generation

### Phase 2: Development Workflow (Weeks 5-8)
- [ ] File watching and change detection (Chokidar)
- [ ] Smart wrapper with auto-update capabilities
- [ ] Incremental update system (parameters, commands, help)
- [ ] Watch mode for continuous development
- [ ] Development vs production mode switching

### Phase 3: Wrapping Engine (Weeks 9-12)
- [ ] Intelligent project analysis
- [ ] Dependency detection and management
- [ ] Wrapper generation for existing projects
- [ ] Advanced communication protocols
- [ ] Error handling and recovery

### Phase 4: Advanced Features (Weeks 13-16)
- [ ] Advanced template system
- [ ] Custom template creation
- [ ] Performance optimization
- [ ] Testing framework
- [ ] Documentation generation

### Phase 5: Polish & Launch (Weeks 17-20)
- [ ] User experience improvements
- [ ] Performance tuning
- [ ] Comprehensive testing
- [ ] Documentation and examples
- [ ] npm package publication

## Success Metrics

### Technical Metrics
- **Wrapper Generation Success Rate**: > 95%
- **Performance**: < 500ms startup time
- **Memory Usage**: < 50MB baseline
- **Error Rate**: < 1% for core operations

### User Metrics
- **Time to First Tool**: < 2 minutes
- **User Satisfaction**: > 4.5/5 rating
- **Adoption Rate**: 1000+ downloads in first month
- **Community Growth**: 100+ GitHub stars in first quarter
- **Development Speed**: 10x faster iteration with auto-sync

### Business Metrics
- **Market Penetration**: 5% of Python CLI tool developers
- **Revenue Potential**: $50K+ annual recurring revenue
- **Partnership Opportunities**: 10+ enterprise partnerships
- **Open Source Impact**: 100+ community contributions

## Risk Assessment

### Technical Risks
1. **Complexity**: Framework may become too complex for simple use cases
   - **Mitigation**: Modular design, clear separation of concerns
2. **Performance**: Communication overhead may impact tool performance
   - **Mitigation**: Protocol optimization, caching, lazy loading
3. **Compatibility**: Version conflicts between Python and Node.js ecosystems
   - **Mitigation**: Comprehensive testing, version pinning

### Market Risks
1. **Competition**: Existing tools may add similar features
   - **Mitigation**: Rapid iteration, unique value propositions
2. **Adoption**: Developers may prefer existing workflows
   - **Mitigation**: Clear benefits, excellent documentation, examples
3. **Maintenance**: Ongoing support for multiple Python/Node.js versions
   - **Mitigation**: Automated testing, community contributions

## Future Roadmap

### Version 1.1 (Q2 2025)
- Plugin system for extensibility
- Advanced template customization
- Performance monitoring and analytics
- Enterprise features and support

### Version 1.2 (Q3 2025)
- Cloud deployment integration
- Advanced debugging tools
- Visual project inspector
- Team collaboration features

### Version 2.0 (Q4 2025)
- AI-powered project analysis
- Automated optimization suggestions
- Multi-language support (Go, Rust, etc.)
- Enterprise-grade security features

## Smart Wrappers & Auto-Update

### Auto-Detection Capabilities
The generated Node.js wrappers include intelligent features that automatically detect and adapt to changes in your Python script:

#### 1. Change Detection
```javascript
// Smart wrapper automatically detects Python script changes
class SmartWrapper {
    async run(args) {
        if (await this.needsUpdate()) {
            console.log('Python script updated, regenerating wrapper...');
            await this.regenerateWrapper();
        }
        return this.executePython(args);
    }
}
```

#### 2. Incremental Updates
- **Parameter Sync**: Only update command-line parameters
- **Command Sync**: Only update command structure
- **Help Sync**: Only update help text and documentation
- **Full Sync**: Complete wrapper regeneration

#### 3. Development Mode
```bash
# Continuous development with auto-sync
pywrap dev reportbuilder --watch

# Automatically:
# - Monitors Python script changes
# - Updates Node.js wrapper
# - Restarts wrapper process
# - Shows live development logs
```

### Benefits of Smart Wrappers
1. **Zero Manual Updates**: Wrapper stays in sync automatically
2. **Fast Development**: No need to manually regenerate wrappers
3. **Production Ready**: Build optimized wrappers for production
4. **Flexible**: Choose update granularity based on needs

## Conclusion

PyWrap represents a significant opportunity to bridge the gap between Python and Node.js ecosystems while providing developers with powerful tools for creating professional CLI applications. The framework's intelligent analysis, dual-mode operation, emphasis on Typer and Rich integration, and **living development workflow** positions it as a unique solution in the developer productivity space.

By focusing on **simplicity for 90% of use cases** and **advanced features when needed**, PyWrap can become the standard tool for Python CLI development and distribution, ultimately improving the developer experience across both Python and Node.js communities.

**Key Innovation**: PyWrap is not just a wrapper generator - it's a **living development framework** that grows with your project.

---

## Appendix

### A. Technology Stack
- **Python**: Typer, Rich, Poetry, uv
- **Node.js**: TypeScript, Commander.js, Chalk
- **Communication**: ZeroMQ, gRPC, WebSocket
- **Testing**: pytest, Jest, Playwright
- **Documentation**: Sphinx, JSDoc, Storybook

### B. Competitor Analysis
- **Click**: Python-only, no npm integration
- **Fire**: Limited CLI capabilities, no wrapping
- **python-shell**: Basic Node.js integration, no framework
- **PyInstaller**: Binary packaging, no npm distribution

### C. Market Research
- **Python CLI Tools**: 10,000+ packages on PyPI
- **npm CLI Tools**: 50,000+ packages on npm
- **Developer Tools Market**: $15B+ annually
- **CLI Tool Adoption**: 80% of developers use CLI tools daily
