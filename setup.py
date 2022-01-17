from setuptools import setup, find_packages


setup(
    name='quantperf',
    version='1.0.3',
    description='QuantPerf',
    packages=find_packages(),
    install_requires=[
        'pandas',
    ]
)
