"""Setup script for iris_classifier package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="iris-classifier",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive machine learning package for Iris flower classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pyenthusiasts/Iris-Flower-Classification",
    project_urls={
        "Bug Tracker": "https://github.com/pyenthusiasts/Iris-Flower-Classification/issues",
        "Documentation": "https://github.com/pyenthusiasts/Iris-Flower-Classification",
        "Source Code": "https://github.com/pyenthusiasts/Iris-Flower-Classification",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "iris-classifier=iris_classifier.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "machine-learning",
        "classification",
        "iris-dataset",
        "scikit-learn",
        "data-science",
        "ml",
        "ai",
    ],
)
