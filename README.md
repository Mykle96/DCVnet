# DVFnet

This repository contains the source code for our master's thesis at the Norwegian University of
Science and Technology.

The code renderes a pipline for calculating the pose of an object from a singel RGB-picture using their mask and a unit vector fields to approximate each keypoint linked to the respective object.

## Installation

### 1. Virtual Environment

First make sure you have initiated a virtual environment. Anaconda was used for this project and can be initiated by running the following commands:

(Make sure you have installed Anaconda first: [Anaconda installer](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) )

```Bash
conda -V #Check that conda is installed and is in your PATH
conda update conda #Check for latest conda
conda create -n yourenvname python=3.8.5 anaconda #Create your virtual environment - with prefered python
conda activate yourenvname
```

Once your virtual environment is activated, install the required packages by either:

```Bash
pip3 install -r requirements
#or
coonda install -r requirements
```
