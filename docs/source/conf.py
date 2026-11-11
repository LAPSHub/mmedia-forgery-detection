import os
import sys

# -- Path setup --------------------------------------------------------------
# Permite que o Sphinx encontre os scripts na raiz do projeto
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'Multimodal Media Forgery Detection'
copyright = '2026, Lucas Nícolas Azevedo Cruz'
author = 'Lucas Nícolas Azevedo Cruz'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Gera docs a partir das Docstrings
    'sphinx.ext.viewcode',     # Adiciona link para ver o código fonte
    'sphinx.ext.napoleon',     # Suporte para Google/NumPy style docstrings
    'sphinx.ext.githubpages', 
]

# --- MOCK IMPORTS (ESSENCIAL PARA O FACE2) ---
# Isso impede que o build falhe por falta de bibliotecas pesadas no servidor
autodoc_mock_imports = [
    "torch", 
    "torchaudio", 
    "mtcnn", 
    "cv2", 
    "moviepy", 
    "tqdm", 
    "numpy", 
    "scipy", 
    "sklearn",
    "matplotlib"
]

# Configurações do Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
try:
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
except ImportError:
    html_theme = 'alabaster'

html_static_path = ['_static']

# Garante que o conteúdo seja lido na ordem em que aparece no código
autodoc_member_order = 'bysource'
