import os

def get_env(name, default, cast=None):
    value = os.environ.get(name, default)
    return value if not cast else cast(value)
