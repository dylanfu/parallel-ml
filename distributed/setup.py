from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  'tensorflow>=1.13',
]

setup(
    name='trainer',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Flower Image Classification model'
)
