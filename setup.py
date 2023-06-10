import setuptools

setuptools.setup(
    name='strawberrypy',
    version='0.2.0', 
    author='Roberta Favata and Nicolas Baù and Antimo Marrazzo',   
    author_email='favata.roberta@gmail.com',
    description='Python package for calculation of topological invariants through single-point formulas and local markers',
    license='MIT License',
    packages=setuptools.find_packages(),
    install_requires=['pythtb==1.8.0',
                      'tbmodels==1.4.3',
                      'opt-einsum==3.3.0',],
)