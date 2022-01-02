import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="handwritten-signature-verification",
    version="0.0.4",
    author="Yash Gupta",
    author_email="eryash15@gmail.com",
    description="Argument passed are image1, image2 & modelname(VGG16, ResNet, AlexNet) to return match/unmatch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eryash15/handsignverify",
    project_urls={
        "Bug Tracker": "https://github.com/eryash15//handsignverify/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)