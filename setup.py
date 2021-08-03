import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="bikit",
    version="0.1.0-alpha",
    author="Philipp J. Roesch, Johannes Flotzinger",
    author_email="philipp.roesch@unibw.de, johannes.flotzinger@unibw.de",
    description="Bridge Inspection Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phiyodr/bridge-inspection-toolkit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition"
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.csv", "*.json"]
    })
