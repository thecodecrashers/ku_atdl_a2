from setuptools import setup, find_packages

setup(name='snn',
      version='0.2',
      description='PAC-Bayes bound optimization for stochastic neural networks (PyTorch port)',
      url='',
      author='',
      author_email='',
      license='Apache-2.0',
      packages=find_packages(),
      python_requires='>=3.9',
      install_requires=[
            'torch>=2.0',
            'numpy>=1.24'
      ],
      zip_safe=False)
