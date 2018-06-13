# Check that the packages are installed.
from pkgutil import iter_modules
import sys


def check_import(packagename):
    if packagename in (name for _, name, _ in iter_modules()):
        return True
    else:
        return False

assert sys.version_info.major >= 3 and sys.version_info.minor >= 6, 'Please install Python 3.6!'

packages = ['jupyter', 'pymc3', 'seaborn', 'matplotlib', 'numpy', 'scipy',
            'pandas', 'tqdm', 'jupyterlab']

all_passed = True

for p in packages:
    assert check_import(p),\
        '{0} not present. Please install via pip or conda.'.format(p)

if all_passed:
    print('All checks passed. Your environment is good to go!')
