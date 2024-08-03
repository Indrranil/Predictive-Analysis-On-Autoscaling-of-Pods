# setup.py
from setuptools import find_packages, setup

setup(
    name='k8s_auto_scaling_project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'pyyaml',
        'joblib',
    ],
    entry_points={
        'console_scripts': [
            'train_model=scripts.train_model:main',
            'evaluate=scripts.evaluate:main',
            'make_prediction=scripts.make_prediction:main',
        ],
    },
)
