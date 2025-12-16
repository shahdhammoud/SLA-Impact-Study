"""
Setup script for the research framework.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="structural-learning-impact",
    version="1.0.0",
    author="Research Team",
    description="Framework for assessing structural learning impact on generative model rankings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shahdhammoud/Assessing-the-impact-of-structural-learning-algorithms-on-the-results-of-comparing-generative-models",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "preprocess-data=scripts.01_preprocess_data:main",
            "train-model=scripts.02_train_model:main",
            "tune-model=scripts.03_tune_model:main",
            "generate-synthetic=scripts.04_generate_synthetic:main",
            "learn-structure=scripts.05_learn_structure:main",
            "evaluate-model=scripts.06_evaluate:main",
            "compare-rankings=scripts.07_compare_rankings:main",
        ],
    },
)
