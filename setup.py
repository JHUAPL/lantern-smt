import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lantern",
    version="0.1.0",
    author="Joshua Brule",
    author_email="joshua.brule@jhuapl.edu",
    description="Tools to encode PyTorch modules as Z3 constraints",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.jhuapl.edu/brulejt1/lantern",
    packages=setuptools.find_packages(exclude=["tests", "example"]),
    python_requires=">=3",
    install_requires=["torch",
                      "z3-solver"]
)
