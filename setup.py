from setuptools import setup, find_packages

setup(
    name='handsignverify',
    version=0.1,
    description='Handwritten Signature Verification with Keras framework',
    url='https://github.com/eryash15/handsignverify',
    author='Yash Gupta',
    author_email='eryash15@gmail.com',
    license='MIT',
    keywords=['keras','signature','deeplearning'],
    install_requires=['numpy>=1.9.1', 'scipy>=0.14', 'h5py', 'pillow', 'keras','six>=1.9.0', 'pyyaml']
)
