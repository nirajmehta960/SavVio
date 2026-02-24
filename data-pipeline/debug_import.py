import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dags"))

try:
    import src.utils
    print("Success! src.utils is located at:", src.utils.__file__)
except Exception as e:
    print("Error importing src.utils:", type(e), e)
    if 'src' in sys.modules:
        print("sys.modules['src']:", sys.modules['src'])
        print("__file__:", getattr(sys.modules['src'], '__file__', 'No __file__'))
        print("__path__:", getattr(sys.modules['src'], '__path__', 'No __path__'))

