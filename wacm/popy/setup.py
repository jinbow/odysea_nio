from setuptools import setup

setup(name='popy',
      version='1.0',
      description='Jinbo Wang\'s software'
      author='Jinbo Wang',
      author_email='jinbow@gmail.com',
      packages=['popy'],  #same as name
      install_requires=['numpy','scipy','xarray'], #external packages as dependencies
      )
