from setuptools import setup, find_packages

setup(
    name="yife-startup-prediction",
    version="1.0.0",
    author="Siddharth Gupta, Pratham Namdev, Shubham Nagar, Sunny Kumar, Anjali Deshwal",
    author_email="guptasiddharth2409@gmail.com",
    description="YIFE: YC-Inspired Feature Engineering for Startup Success Prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/guptasiddharth2409/yife-startup-prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "shap>=0.44.0",
        "torch>=2.1.0",
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
