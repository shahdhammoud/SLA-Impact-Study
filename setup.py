from setuptools import setup, find_packages

setup(
    name='sla-benchmark',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'sla-benchmark=sla:main',
        ],
    },
)
