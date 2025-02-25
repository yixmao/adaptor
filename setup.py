from setuptools import setup, find_packages

setup(
    name='adaptor',
    version='0.1.0',
    description='AdapToR: Adaptive Topological Regression for quantitative structure-activity relationship modeling',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Yixiang Mao',
    author_email='yixmao@ttu.edu',
    url='https://github.com/yixmao/adaptor',
    license='MIT',            
    packages=find_packages(include=['adaptor', 'adaptor.*']),
    install_requires=[
        'numpy',
        'scipy',
        'pandas', 
        'scikit-learn',
        'fastparquet',
        'rdkit'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Specify Python version compatibility
)