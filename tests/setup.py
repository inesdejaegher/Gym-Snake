from setuptools import setup, find_packages

setup(name='gym_snake',
      version='0.0.1',
      author="Satchel Grant",
      packages=find_packages(include=['gym_snake', 'gym_snake.*']),
      install_requires=['gym', 'numpy', 'matplotlib'],
      python_requires='>=3',
)
