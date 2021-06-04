from setuptools import setup

setup(
   name='hugs2021',
   version='0.1',
   description='Toy examples for HUGS2021',
   author='Malachi Schram',
   author_email='schram@jlab.org',
   packages=['hugsdata'],  #same as name
      install_requires=['gym','keras','pandas',
                        'numpy','matplotlib','seaborn',
                        'sklearn','scipy',
                        'tqdm','jupyter']  
)
