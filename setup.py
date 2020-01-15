from setuptools import setup, find_packages

setup(
    name="pam",
    version="0.1.0",
    author="Bryce Kalmbach",
    author_email="brycek@uw.edu",
    url="https://github.com/jbkalmbach/pam",
    packages=find_packages(),
    description="Photo-Z Data Augmentation",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
)