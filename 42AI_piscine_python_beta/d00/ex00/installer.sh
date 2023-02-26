#!/bin/bash
#this script installs python

#export PATH=/goinfre/tduquesn/python3/bin:$PATH to use the newly installed version and obtain permissions
#which pip or which python to check version used

curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o miniconda_installer.sh
sh miniconda_installer.sh -b -p /goinfre/tduquesn/python3 #/Users/tduquesn/goinfre/python3
export PATH=/goinfre/tduquesn/python3/bin:$PATH
rm miniconda_installer.sh
