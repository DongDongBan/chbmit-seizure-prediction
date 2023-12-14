from pyedflib import EdfReader

__all__ = [
    'EdfReaderWrapper'
]

class EdfReaderWrapper(EdfReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        super().close()
