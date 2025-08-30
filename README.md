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

## The CLASSICS package
One of the provided cross sections is based on the CLASSICS package, with a
repo [here](https://github.com/kahlhoefer/CLASSICS). Our intention is use this
as a dependency, however, CLASSICS does not currently provide any means to 
install it (like a `pyproject.toml` or `setup.py`). Instead the expected use
appears to be to copy the entire folder into your current working directory and
go from there. This is what we will do **temporarily** until the installation
mechanism is updated. And by that, I mean include the entire CLASSICS folder
as a subdirectory of the `sidm-vdsigmas` package.

# Tutorials
See the `tutorials` folder for some python notebooks showing how to
use package.
