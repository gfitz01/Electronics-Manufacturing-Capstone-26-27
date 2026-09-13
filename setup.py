from setuptools import setup, find_packages

package_name = 'DeepManufacturing'

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

classifiers = [
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3 :: Only',
    'Intended Audience :: Education']

setup(name=package_name,
      version='0.0.8',
      description='AI and Machine Learning for manufacturing related datasets',
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      author='Amir Barati Farimani',
      author_email='barati@andrew.cmu.edu',
      license='MIT',
      classifiers=classifiers,
      keywords=' ',
      # 3.9 is the floor because the torch/torchvision versions this now
      # targets dropped support for anything older.
      python_requires='>=3.9, <4',
      install_requires=[
          # torch and torchvision were previously missing from this list
          # entirely, so `pip install DeepManufacturing` produced an install
          # that could not import any deep-learning model.
          'torch>=2.0',
          'torchvision>=0.15',
          'numpy>=1.23',
          'scipy>=1.9',
          'matplotlib>=3.6',
          'xgboost>=1.7',
          'scikit-learn>=1.2',
          'timm>=0.9',
          'Pillow>=9.0',
          'requests>=2.28',
      ],
      extras_require={
          # Only needed for the Google Drive download path, which is now
          # optional -- see ManufacturingNet/datasets/datasets.py.
          'drive': ['gdown>=5.1'],
      },
      packages=find_packages(),
      zip_safe=False)
