import setuptools

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mshap",
    version="0.0.1",
    author="Diadochokinetic",
    author_email="diadochokinetic@googlemail.com",
    description="A Python port of mshap two interpret combined model outputs.",
    python_requires=">=3.10",
    long_description=long_description,
    packages=setuptools.find_packages(),
    install_requires=requirements,
)
