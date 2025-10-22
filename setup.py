#!/usr/bin/env python3
"""
Setup script for Sound_Vision project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sound-vision",
    version="0.1.0",
    author="sudo-abeaver",
    description="A computer vision project focused on object detection using YOLOv8 and advanced vision models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sudo-abeaver/Sound_Vision",
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
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "clip": ["transformers", "torch"],
        "detic": ["detectron2"],
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "sound-vision-detect=scripts.detect_with_yolov8:main",
            "sound-vision-webcam=scripts.detic_webcam:main",
        ],
    },
)
