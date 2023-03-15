from distutils.core import setup, Extension


spa_module = Extension('_sunpos',
                           sources=['sunpos_wrap.c', 'SunPos.c'],
                           headers=['SunPos.h'],
                           )

setup (name = 'sunpos',
#       ext_package='SPAmodule',
       version = '1.1',
       author      = "K.A. Shmirko",
       License = 'GNU',
       Platform = 'Any',
       description = """Sun position calculator""",
       ext_modules = [spa_module],
       py_modules = ["sunpos"],
       )
