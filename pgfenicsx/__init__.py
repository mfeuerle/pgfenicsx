# Moritz Feuerle, 2025
r""" 
Classes
-----------------
.. autosummary::
   :toctree: _generated/
   
   DirichletBC
   
Functions
-----------------
.. autosummary::
   :toctree: _generated/ 
   
   dirichletbc
   collect_dirichletbcs


.. sectionauthor:: Moritz Feuerle, 2022
"""
# generating the list of functions / classes / etc. above automatically might be bossible with https://stackoverflow.com/a/18143318


from ._pgfenicsx import *

# prevent that all that clutter below end up in __all__
__all__ = [s for s in dir() if not s.startswith('_')]



# Hacky workaround for TypeAlias not being shown in the docs
# Might be fixed in Sphinx v9.0, see the upcoming option .. autotype:: and https://github.com/sphinx-doc/sphinx/pull/13808
# from typing import TypeAlias as _TypeAlias
# from typing import TypeAliasType as _TypeAliasType

# import dolfinx

# for x in dir():
#    if isinstance(eval(x),_TypeAliasType):
#       exec("%s: _TypeAlias = %s" % (x,str(eval(x).__value__)))
