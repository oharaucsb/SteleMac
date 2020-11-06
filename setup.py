from setuptools import find_namespace_packages, setup

# to understand how setuptools and many following commands work, see
# https://setuptools.readthedocs.io/en/latest/setuptools.html

setup(
    # ***distribution metadata***

    url='https://github.com/C-S-Cannon/Stele',
    author='Cameron Cannon',
    author_email='c_cannon@ucsb.edu',
    license='MIT',
    description="Monolithic python distribution to house ITST's HSG code.",


    # ***package setup***

    # package name
    name='Stele',
    # what version, pre release codes are 'anything' < 'final'
    # equivelat post finals, c = pre = preview = rc
    # post release codes are 'anything' > 'final'
    # thus 1.0 < 1.1a < 1.1pre < 1.1 < 1.1finally < 1.1g
    version='0.1a2',
    # what and where are the packages, conforms to src/ layout
    # calls setuptools member to automate this, allows setup.py auto update
    packages=find_namespace_packages(where='src'),
    # source directory for the given package
    package_dir={'': 'src'},
    # should data inside package be included
    include_package_data=True,
    # can package be installed from Zip?
    zip_safe=False,
    # What is the runPEPS command in use at this line?
    # entry_points={'console_scripts': ['runPEPS = PEPS.command_line:main']},


    # ***requirements***

    # required version of python is 3.3 or greater as defined in PEP 440
    # necessary since this is a namespace package and 2.7 is EOL
    python_requires='>=3.5',
    # packages required to run the setup script itself
    setup_requires=[],
    # URLS to be searched for setup_requires dependencies
    dependency_links=[],
    # packages required for the distribution package to function
    install_requires=[
        'glob2>=0.6',
        'joblib>=0.13.1',
        'jsonschema>=2.6.0',
        'matplotlib>=2.2.2',
        'numpy>=1.14.3',
        'PyQt5==5.14.0',
        'pyqtgraph==0.10.0',
        'scipy>=1.1.0'
        ],
    # dependencies that are necessary for additional usages of the distro
    # install extras via brackets concattenated to the package name in pip
    # EX "pip install Stele[dev]"
    extras_requires={
        # dependencies for development of the distribution package
        'dev': [
            # style collection used for this distribution, optional dependency
            'pylama>=7.7.1',
            # automatic documentation support via doclines, optional dependency
            'Sphinx>=3.2.1',
            # darkmode HTML theme for sphinx generated documentation
            'sphinx_pdj_theme=>0.2.1'
            ]
        # TODO: Look at utilizing extras for correllated modules that are only
        #   required in rare cases but bring many new dependency requirements.
        }
)
