# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os, sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'StraWBerryPy'
copyright = '2023, Roberta Favata, Nicolas Baù and Antimo Marrazzo'
author = 'Roberta Favata, Nicolas Baù and Antimo Marrazzo'
release = '0.3.2'
language = 'en'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.viewcode',
	'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
highlight_language = 'python3'
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# from: https://framagit.org/coslo/template-docs

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
#html_css_files = ['custom.css']
pygments_style = 'friendly'

html_logo = "_static/media/logo.png"
html_title = "StraWBerryPy"
html_favicon = "favicon.ico"

html_theme_options = {
    "show_prev_next": False,
    "repository_url": "https://github.com/strawberrypy-developers/strawberrypy.git",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "logo": {
        "image_dark": html_logo,
        "image_light": html_logo,
        "text": "<h1>StraWBerryPy</h1>\n<p>Single-poinT invaRiAnts and local</p>\n<p>markers for Wannier Berriologies</p>\n<p>in Python</p>",
    },
    "toc_title": "&nbsp;On this page",
    "use_repository_button": True,
    "use_issues_button": False,
    "use_edit_page_button": False,
    "use_download_button": True,
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 40,
    "footer_content_items": ["copyright.html"],
    "home_page_in_toc": True,
}

html_sidebars = {
    "**": ["navbar-logo.html", "search-field.html", "sbt-sidebar-nav.html"]
}