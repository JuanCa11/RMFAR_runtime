from setuptools import find_packages, setup

# Library dependencies
INSTALL_REQUIRES = [
    "pandas==0.24.2",
    "scikit-learn==0.20.3",
    "mlxtend==0.18.0",
    "numpy==1.16.2"
]

setup(
    name='fuzzy_ar',
    version="0.0.1",
    description="Fuzzy Association Rules Utility",
    author="Juan Camilo Montaña Granados",
    author_email="juanc_montana@javeriana.edu.co",
    url="git@github.com:JuanCa11/fuzzy-association-rules.git",
    packages=find_packages(),
    license='GPLv3+',
    install_requires=INSTALL_REQUIRES,
)