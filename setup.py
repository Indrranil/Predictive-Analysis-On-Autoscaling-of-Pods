from setuptools import setup, find_packages

setup(
    name='predictive_autoscaling',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'pyyaml',
    ],
)
