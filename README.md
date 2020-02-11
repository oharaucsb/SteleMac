# Stele
Monolithic python distribution to eventually house ITST code base.

A few things of note. The eventual goal is for this to require conformity
  to push to the repository. Currently pep8, Pyflakes, McCabe, isort compliant
  based on the Pylama linter for atom.

I recognize this may well be obnoxious to ensure code compliance, it
  certainly has been for me. I would however strongly encourage you to retain
  this feature, for it is likely one of the greatest safeguards against such
  issues that required the entire code based be retyped in the future.

Stele uses namespace packaging for the distribution packages (distros). This
  requires python 3.3+, which the setup file will check for. If any issues are
  encountered, this fact may be worth noting. It helped me at least.
