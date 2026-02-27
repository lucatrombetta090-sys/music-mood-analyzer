"""
Recipe numpy personalizzato — Fix URL download
p4a v2024.01.21 usa pypi.python.org (deprecato/morto).
Questo recipe locale sovrascrive l'URL con files.pythonhosted.org
"""
from pythonforandroid.recipes.numpy import NumpyRecipe as _Base


class NumpyRecipe(_Base):
    # Stessa versione della recipe ufficiale p4a v2024.01.21
    version = '1.24.4'
    # URL corretto su files.pythonhosted.org (pypi.python.org è deprecato)
    url = ('https://files.pythonhosted.org/packages/source/n/numpy/'
           'numpy-{version}.zip')


recipe = NumpyRecipe()
