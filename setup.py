from setuptools import find_packages, setup

setup(
    name='pytorch',
    version='0.0.1',
    author='Anant Srivastava',
    author_email="itanart333@gmail.com",
    install_requires=[
        'openai',
        'matplotlib',
        'streamlit',
        'torchvision',
        'Pytorch'
    ],
    packages=find_packages(),
)