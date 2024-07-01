from setuptools import setup, find_packages

setup(
    name='torch-trainer',
    version='1.0.0',
    license="MIT",
    description='A simple library to train pytorch models.',
    long_description=open('README.md').read(),
    author='gdamms',
    author_email='damguillotin@gmail.com',
    url='https://www.github.com/gdamms/torch-trainer',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'rich',
        'torch',
        'tensorboard',
    ],
)