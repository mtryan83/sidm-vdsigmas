# Intro

This is a basic README for the sidm-vdsigmas package.

# Installation

This may eventually end on PyPi or conda-forge, but for now, you'll 
need to clone this repository:
```sh
git clone https://github.com/mtryan83/sidm-vdsigmas.git
```
or (if you have ssh keys set up)
```sh
git clone git@github.com:mtryan83/sidm-vdsigmas.git
```
and then install locally:
```sh
cd sidm-vdsigmas
python -m pip install -e .
```

Note that we've installed the package as an editable (the `-e` flag)
in case you want to add your own cross sections or other functions.

# Tutorials
See the `tutorials` folder for some python notebooks showing how to
use package.
