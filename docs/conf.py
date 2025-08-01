# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# path
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mssm'
copyright = '2024, Joshua Krause'
author = 'Joshua Krause'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              "myst_nb"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- myst_nb configuration ---------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/configuration.html
nb_execution_timeout = 500
nb_execution_show_tb = True
nb_scroll_outputs = True

myst_enable_extensions = [
                          "dollarmath"
                         ]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {

    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 8
}


html_static_path = ['_static']
