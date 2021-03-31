from setuptools import find_packages, setup

setup(
    name='ml_project',
    packages=find_packages(),
    version='0.1.0',
    description='This is the first homework project',
    author='Sergey Polyanskikh',
    install_requires=[
        "click==7.1.2",
        "python-dotenv>=0.5.1",
        "scikit-learn==0.24.1",
        "dataclasses==0.8",
        "pyyaml==3.11",
        "marshmallow-dataclass==8.3.0",
        "pandas==1.1.5",
    ],
    license='MIT',
)
