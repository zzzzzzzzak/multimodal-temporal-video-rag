from setuptools import setup, find_packages

setup(
    name="multimodal-temporal-video-rag",
    version="1.0.0",
    description="Multimodal temporal video retrieval and moment grounding engine using dual encoders and cross-attention re-ranking.",
    author="Siddiqui Zakir Ali",
    author_email="zak.siddiqui16@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.22.0",
        "scipy>=1.10.0",
        "opencv-python-headless>=4.8.0",
        "pillow>=9.5.0",
        "matplotlib>=3.7.0",
        "streamlit>=1.28.0",
    ],
)
