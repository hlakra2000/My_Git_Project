from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements= [req.replace("\n","") for req in requirements]
    return requirements
setup(
    name='Traffic-congestion',
    version='0.0.1',
    author='Harsh',
    author_email='hlakra2000@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('Requirements.txt')
)