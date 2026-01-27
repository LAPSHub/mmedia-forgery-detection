import os
import sys

# -- Path setup --------------------------------------------------------------
# Isso permite que o Sphinx encontre seus scripts face2.py e data_augmentation.py
# que estão na raiz do projeto (dois níveis acima deste arquivo)
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'Multimodal Pipeline'
copyright = '2024, Seu Nome'
author = 'Seu Nome'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Gera docs a partir das Docstrings
    'sphinx.ext.viewcode',     # Adiciona link para ver o código fonte
    'sphinx.ext.napoleon',     # Suporte para Google/NumPy style docstrings
    'sphinx.ext.githubpages',  # Útil se você for publicar no GitHub Pages
]

# Configurações do Napoleon para garantir que ele leia bem seus detalhes
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# O tema 'sphinx_rtd_theme' é o padrão profissional (azul/cinza)
# Se não o tiver instalado, use: pip install sphinx_rtd_theme
try:
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
except ImportError:
    html_theme = 'alabaster' # Tema padrão caso o outro não esteja instalado

html_static_path = ['_static']
