#!/bin/bash
#should allow installation via the command `bash build.sh && pip install ./dist/42ai-1.0.0.tar.gz`
#https://dzone.com/articles/executable-package-pip-install
#https://packaging.python.org/
#https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/creation.html

python3 setup.py sdist --formats gztar