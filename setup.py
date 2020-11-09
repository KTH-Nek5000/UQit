import setuptools

with open("README_pypi.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="UQit", 
    version="1.0.2",
    author="Saleh Rezaeiravesh",
    author_email="salehr@kth.se",
    description="A Python Package for Uncertainty Quantification (UQ) in Computational Fluid Dynamics (CFD)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KTH-Nek5000/UQit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
