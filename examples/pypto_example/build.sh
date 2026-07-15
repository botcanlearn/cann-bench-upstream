#!/bin/bash
set -e
python setup.py bdist_wheel
rm -rf *.egg-info
