# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = '<project-name>'
copyright = '2024, Sam Griesemer'
author = 'Sam Griesemer'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary", # enables a directive to be specified manually that gathers
                              # module/object summary details in a table
    "sphinx.ext.viewcode",    # allow viewing source in the HTML pages
    "myst_parser",            # only really applies to manual docs; docstrings still need RST-like
    "sphinx.ext.napoleon",    # enables Google-style docstring formats
    "sphinx_autodoc_typehints", # external extension that allows arg types to be inferred by type hints
]
autosummary_generate = True
autosummary_imported_members = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
#html_sidebars = {
#    '**': ['/modules.html'],
#}

