from setuptools import setup, find_packages

setup(
    name="mvrag",
    version="0.1.0",
    description="MV-RAG: Retrieval-Augmented Multiview Diffusion",
    author="Yosef Dayani",
    author_email="yoseph.dayani@mail.huji.ac.il",
    url="https://github.com/yosefdayani/MV-RAG",
    packages=find_packages(include=["mvrag", "retrievers", "modules", "mvrag.*", "retrievers.*", "modules.*"]),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "mvrag=main:main",
        ],
    },
)
