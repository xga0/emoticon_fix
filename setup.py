import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="emoticon_fix",
    version="0.0.4",
    author="Sean Gao",
    author_email="seangaoxy@gmail.com",
    description="Transform emoticon to text, e.g., :) => Smile.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xga0/emoticon_fix",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    python_requires='>=3.6',
    install_requires=[
        'nltk>=3.5',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'mypy>=0.800',
            'flake8>=3.9',
        ],
    },
)