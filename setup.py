from setuptools import setup, find_packages

setup(
    name='micro-scgpt',
    version='0.0.1',
    url='https://github.com/Risitop/micro-scgpt',
    author='Risitop',
    author_email='aziz.fouche@gmail.com',
    description='A lightweight GPT for single-cell data',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "scanpy", 
        "torch"
    ]
)