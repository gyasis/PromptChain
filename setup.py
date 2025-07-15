from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="promptchain",
    version="0.2.4",
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
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "promptchain": ["prompts/**/*.md"],
        "promptchain.ingestors": ["**/*.py"],
        "promptchain.extras": ["**/*.py"],
    },
) 