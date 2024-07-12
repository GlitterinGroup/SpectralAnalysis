from setuptools import find_packages, setup

setup(
    name='SpectralAnalysis',
    version='0.2.0',
    description='A toolkit for spectral analysis',
    author='Glitterin',
    author_email='your.email@example.com',
    url='https://github.com/GlitterinGroup/SpectralAnalysis',  
    packages=find_packages(include=['spectral_analysis', 'spectral_analysis.*']),
)

