from pathlib import Path
import urllib.request
from urllib.parse import urlparse

script_cache = Path(__file__).parent.joinpath("tmp")
script_cache.mkdir(exist_ok=True)

# TODO