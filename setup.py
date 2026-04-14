from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Computer Vision-based waste classification system using EfficientNet with MLflow tracking, DVC pipelines, and FastAPI deployment. Designed for scalable, reproducible, and production-ready ML workflows.',
    author='Shakil Ur Rehman',
    license='MIT',
)
