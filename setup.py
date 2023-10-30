import setuptools

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mshap",
    version="0.2.2",
    author="Diadochokinetic",
    author_email="diadochokinetic@googlemail.com",
    description="A Python port of R package mshap to interpret combined model outputs.",
    python_requires=">=3.10",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=requirements,
)
