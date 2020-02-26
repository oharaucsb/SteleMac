# Stele
Monolithic python distribution package to eventually house ITST code base.

A few things of note.

The eventual goal is for this to require conformity
  to push to the repository. Current idea is pep8, Pyflakes, McCabe, isort
  compliant based on the Pylama linter for atom.

Stele uses namespace packaging for the distribution packages (distros). This
  requires python 3.3+, which the setup file will check for. If any issues are
  encountered, this fact may be worth noting. It helped me at least.

Packages go within the src directory.

<<<<<<< HEAD
Test code goes within test.

Development code for new features goes in dev until being incorporated within
=======
Test code going within test.

development code for new features goes in dev until being incorporated within
>>>>>>> f93a6affd5ffd93359c3811a7761bc9ea08d4c46
src.
