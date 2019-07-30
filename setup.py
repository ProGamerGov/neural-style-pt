import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="neural-style",
    version="",
    author="ProGamerGov",
    description="A PyTorch implementation of artistic style transfer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    scripts=['neural-style'],
    url="https://github.com/ProGamerGov/neural-style-pt/",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)