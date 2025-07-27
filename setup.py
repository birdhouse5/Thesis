#!/usr/bin/env python3
"""
VariBAD Portfolio Optimization - Setup Script
Installs the package and dependencies for cloud deployment
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return requirements

setup(
    name="varibad-portfolio",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Regime-agnostic reinforcement learning for portfolio optimization using variBAD",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/varibad-portfolio",
    
    # Package discovery
    packages=find_packages(),
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Package classification
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    # Keywords for discovery
    keywords="reinforcement-learning, portfolio-optimization, variBAD, finance, machine-learning",
    
    # Entry points for command-line usage
    entry_points={
        "console_scripts": [
            "varibad-train=varibad.scripts.main:main",
        ],
    },
    
    # Additional files to include
    package_data={
        "varibad": [
            "data/*.parquet",
            "config/*.yaml",
        ],
    },
    
    # External data files
    data_files=[
        (".", ["README.md", "requirements.txt"]),
        ("scripts", ["setup_cloud.sh", "git_workflow.sh"]),
    ],
    
    # Development dependencies
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "jupyter>=1.0",
            "matplotlib>=3.3",
            "seaborn>=0.11",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
        ],
        "analysis": [
            "jupyter>=1.0",
            "matplotlib>=3.3",
            "seaborn>=0.11",
            "plotly>=5.0",
            "dash>=2.0",
        ],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/yourusername/varibad-portfolio/issues",
        "Source": "https://github.com/yourusername/varibad-portfolio",
        "Documentation": "https://github.com/yourusername/varibad-portfolio/wiki",
    },
    
    # Include all Python files
    include_package_data=True,
    
    # Zip safe
    zip_safe=False,
)