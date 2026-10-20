"""Run the factory recipe with a proxy environment httpx can parse.

The host injects NO_PROXY containing an IPv6 entry, which httpx turns into an
invalid URL pattern ("Invalid port: ':1]'"). Cleaning it here, before httpx is
imported, keeps the fix local to this workspace instead of editing the factory.
"""
import os
import runpy
import sys

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ.setdefault("PYTHONUTF8", "1")

recipe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "recipes", "scale_swe", "run.py")
sys.argv = [recipe] + sys.argv[1:]
runpy.run_path(recipe, run_name="__main__")
