from pythonforandroid.recipes.sqlite3 import Sqlite3Recipe
from pythonforandroid.util import load_source
import os

class Sqlite3Recipe(Sqlite3Recipe):
    version = '3.42.0'
    url = 'https://www.sqlite.org/2023/sqlite-autoconf-3420000.tar.gz'
    
    def get_recipe_env(self, arch):
        env = super().get_recipe_env(arch)
        env['CFLAGS'] += ' -DSQLITE_ENABLE_COLUMN_METADATA=1'
        return env

recipe = Sqlite3Recipe()
