from cx_Freeze import setup, Executable
import sys

base = None

if sys.platform == 'win32':
    base = None

executables = [Executable("gui.py", base=base)]

packages = ["idna"]
options = {
    'build_exe': {

        'packages': packages,
    },

}

setup(
    name="RheoPlots",
    options=options,
    version="1.0",
    description='RheoPlots 1.0 by Petrus Kirsten.',
    executables=executables
)
