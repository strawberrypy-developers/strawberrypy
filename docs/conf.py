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
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.viewcode',
	'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinxawesome_theme.highlighting'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
highlight_language = 'python3'
master_doc = 'index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# from: https://framagit.org/coslo/template-docs

html_theme = 'alabaster'
html_static_path = ['_static']

if html_theme == 'alabaster':
    pygments_style = 'friendly'
    html_static_path = ['_static/custom.css']
    html_theme_options = {
        'description': 'Single-poinT and local invaRiAnts for Wannier Berriologies in Python',
        'fixed_sidebar': True,
        'sidebar_collapse': True,
        'extra_nav_links': {},
        'gray_2': '#F4F4F4ED',
        'sidebar_width': '250px',
        'body_max_width': 'auto',
        'page_width': '1000px'
    }
    html_logo = "_static/media/logo.png"
    
    html_favicon = "favicon.ico"

    html_sidebars = {
        '**': [
            'about.html',
            'navigation.html',
            'searchbox.html',
            'relations.html',
            'donate.html',
        ]
    }

if html_theme == 'sphinx_rtd_theme':
    import sphinx_rtd_theme
    extensions += [
        'sphinx_rtd_theme',
    ]
    html_theme_options = {
        'display_version': True,
        'vcs_pageview_mode': '',
        # Toc options
        'collapse_navigation': False,
        'sticky_navigation': True,
        'navigation_depth': 4,
        'includehidden': True,
        'titles_only': False
    }

if html_theme == 'furo':
    pygments_style = 'tango'
    # html_static_path = ['_static/furo/']
    # html_css_files = ['custom.css']
    html_theme_options = {
        "light_css_variables": {
            "admonition-title-font-size": "1rem",
            "admonition-font-size": "1rem",
        },
    }
