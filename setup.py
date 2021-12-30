from cx_Freeze import setup, Executable

base = None    

executables = [Executable("Surveillance_Application.py", base=base)]

packages = ["idna"]
options = {
    'build_exe': {    
        'packages':packages,
    },    
}

setup(
    name = "CARS_IRDE",
    options = options,
    version = "1.0.2",
    description = 'Border Surveillance System',
    executables = executables
)