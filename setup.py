import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="emoticon_fix",
    version="0.0.2",
    author="Sean Gao",
    author_email="seangaoxy@gmail.com",
    description="Transform emoticon to text, e.g., :) => Smile.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xga0/emoticon_fix",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)