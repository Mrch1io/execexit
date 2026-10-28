import importlib.metadata

try:
    __version__ = importlib.metadata.version("execexit")
except importlib.metadata.PackageNotFoundError:
    pass
