from setuptools import setup, find_packages

setup(
    name="spiking_gpt_oss",
    version="0.1.0",
    description="A simple spiking wrapper for GPT‑OSS models",
    author="Your Name",
    packages=find_packages(include=["spiking_gpt_oss"]),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "sentencepiece>=0.1.99",
        "huggingface-hub>=0.20.0",
    ],
)