# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os, sys, re

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "StraWBerryPy"
copyright = "2026, Roberta Favata, Nicolas Baù, Christian Loer Llemit, Surajit Manna and Antimo Marrazzo"
author = "Roberta Favata, Nicolas Baù, Christian Loer Llemit, Surajit Manna and Antimo Marrazzo"
release = re.search(
    r".+version[^0-9]+([0-9.]+)", open("../strawberrypy/config.py").read()
).group(1)
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
highlight_language = "python3"
master_doc = "index"

autodoc_typehints = "description"
autodoc_members_order = "bysource"
autoclass_content = "both"
autosummary_generate = True
numfig = True

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
rst_prolog = """
.. role:: python(code)
    :language: python
    :class: highlight
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_css_files = ["custom.css"]
html_js_files = ["fix-scrollbar.js"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
pygments_dark_style = "monokai"
pygments_light_style = "tango"

html_logo = "_static/media/logo.png"
html_title = "StraWBerryPy"
html_favicon = "favicon.ico"

html_theme_options = {
    "show_prev_next": False,
    "repository_url": "https://github.com/strawberrypy-developers/strawberrypy.git",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "logo": {"image_dark": html_logo, "image_light": html_logo, "text": ("StraWBerryPy")},
    "toc_title": "&nbsp;On this page",
    "use_repository_button": True,
    "use_issues_button": False,
    "use_edit_page_button": False,
    "use_download_button": True,
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 5,
    "footer_content_items": ["copyright.html"],
    "home_page_in_toc": False,
}

html_sidebars = {
    "**": [
        "navbar-logo.html",
        "subtitle.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
    ]
}
