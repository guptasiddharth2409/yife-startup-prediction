from setuptools import setup, find_packages

setup(
    name="yife",
    version="1.1.0",
    description="YC-Inspired Feature Engineering for Startup Outcome Classification",
    author="Siddharth Gupta",
    author_email="guptasiddharth2409@gmail.com",
    url="https://github.com/guptasiddharth2409/yife-startup-prediction",
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "shap>=0.44",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "joblib>=1.3",
        "pyarrow>=12.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
