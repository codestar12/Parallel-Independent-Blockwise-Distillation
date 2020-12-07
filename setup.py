from setuptools import setup, find_packages
setup(
    name="train_layer",
    version="0.1",
    packages=find_packages(include=['blockwise', 'blockwise.*'],
                           exclude=['__pycache__'])
)
