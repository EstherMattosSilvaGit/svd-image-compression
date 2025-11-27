from importlib import util
spec = util.spec_from_file_location('pca', 'src/pca_mnist.py')
mod = util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    print('Module loaded OK')
    import sys
    print('numpy in sys.modules:', 'numpy' in sys.modules)
    print('matplotlib in sys.modules:', 'matplotlib' in sys.modules)
except Exception as e:
    print('Import error:', e)
