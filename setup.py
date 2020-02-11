from setuptools import find_namespace_packages, setup

# to understand how setuptools and many following commannds work, see
# https://setuptools.readthedocs.io/en/latest/setuptools.html

setup(

    # ***package setup***
    name='Stele',  # package name
    version='0.1a0',
            # what version, pre release codes are 'anything' < 'final'
            # equivelat post finals, c = pre = preview = rc
            # post release codes are 'anything' > 'final'
            # thus 1.0 < 1.1a < 1.1pre < 1.1 < 1.1finally < 1.1g
    packages=find_namespace_packages(where="src"),
    # what and where are the packages, conforms to src/ layout
    # calls setuptools member to automate this, allows setup.py auto update
    package_dir={"": "src"},
    # source directory for the given package
    include_package_data=True,
    # should data inside package be included
    zip_safe=False,
    # can package be installed from Zip?

    # entry_points={'console_scripts': ['runPEPS = PEPS.command_line:main']},
    # What is the runPEPS command in use at the above line?

    # ***requirements***
    python_requires='>=3.3',
    # required version of python is 3.3 or greater as defined in PEP 440
    # necessary since this is a namespace package and 2.7 is EOL
    setup_requires=[],
    # packages required to run the setup script itself
    dependency_links=[],
    # URLS to be searched for setup_requires dependencies
    install_requires=[],
    # packages required for the package to function

    # ***distribution metadata***
    url='https://github.com/C-S-Cannon/Stele',
    author='Cameron Cannon',
    author_email='c_cannon@ucsb.edu',
    license='GNU General Public',
    description="Monolithic python distribution to eventually house ITST code."
)

# look at utilizing extras for correllated modules that are only required in
# rare cases but bring many new dependency requirements
