from setuptools import setup, find_packages

setup(
    name="text2sql-slm-optimization",
    version="0.1.0",
    description="Text-to-SQL SLM research: Prompt Repetition + RE2 Re-Reading",
    author="Shourya Aswal",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.5.0",
        "transformers>=4.48.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.45.0",
        "datasets>=2.21.0",
        "accelerate>=1.2.0",
        "trl>=0.13.0",
        "sentencepiece>=0.2.0",
        "sqlparse>=0.5.0",
        "pandas>=2.2.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
)
