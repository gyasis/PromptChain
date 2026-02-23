from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies required for all users
core_requirements = [
    "litellm>=1.0.0",           # Unified LLM access
    "python-dotenv>=1.0.0",     # Environment variable management
    "pydantic>=2.0.0",          # Data validation
    "requests>=2.31.0",         # API calls
    "typing-extensions>=4.8.0", # Improved type hints
    "tiktoken>=0.5.0",          # Token counting in execution history
    "rich>=13.0.0",             # Console output in agent_chain
    "httpx>=0.25.0",            # HTTP requests
    "pyyaml>=6.0",              # YAML configuration files
    "jsonschema>=4.17",         # YAML schema validation
    "nest-asyncio>=1.5.0",      # Async support for nested event loops
]

setup(
    name="promptchain",
    version="0.5.0",
    author="Gyasi Sutton",
    author_email="gyasis@gmail.com",
    description="A flexible prompt engineering library for LLM applications and agent frameworks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gyasis/promptchain",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "cli": [
            "textual>=0.83.0",          # Terminal UI
            "click>=8.1.0",             # CLI framework
            "prompt_toolkit>=3.0.0",    # Input handling
        ],
        "dev": [
            # Observability (optional)
            "mlflow>=2.9.0",            # MLflow tracking and observability

            # Testing
            "pytest>=7.4.0",            # Testing framework
            "pytest-asyncio>=0.21.0",   # Async test support
            "pytest-mock>=3.10.0",      # Mocking support

            # Code quality
            "black>=23.12.0",           # Code formatting
            "isort>=5.13.0",            # Import sorting
            "flake8>=6.1.0",            # Linting
            "mypy>=1.7.0",              # Type checking

            # Documentation
            "mkdocs>=1.5.0",            # Documentation generation
            "mkdocs-material>=9.5.0",   # Documentation theme
        ],
        "all": [
            # All optional features
            "textual>=0.83.0",
            "click>=8.1.0",
            "prompt_toolkit>=3.0.0",
            "mlflow>=2.9.0",
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
            "black>=23.12.0",
            "isort>=5.13.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "promptchain=promptchain.cli.main:main [cli]",
        ],
    },
    include_package_data=True,
    package_data={
        "promptchain": ["prompts/**/*.md"],
        "promptchain.ingestors": ["**/*.py"],
        "promptchain.extras": ["**/*.py"],
    },
) 