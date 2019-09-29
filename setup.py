import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neural-style",
    version="0.4",
    author="ProGamerGov",
    description="A PyTorch implementation of artistic style transfer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='neural artistic style neural-style neural-style-pt pytorch style transfer style-transfer',
    scripts=['neural-style'],
    url="https://github.com/ProGamerGov/neural-style-pt/",
    packages=setuptools.find_packages(),
    install_requires=['torch', 'torchvision', 'pillow'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Artistic Software",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
