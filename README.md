# The Office: Analysis of lines for the main characters
![Alt text](docs/imgs/the-office-992x560.jpg?raw=true "The Office")

## Table of content

- [Purpose](#Purpose)
- [Features](#Features)
- [Setup](#Setup)
  - [Clone repo](#Clone-repo)
  - [Conda env](#Conda-env)
  - [Code formatting](#Formatting)
  - [Testing](#Testing)

## Purpose
I was looking for a nice data science project - nice as in novel, helpful to humanity, ... - but didn't bother doing the classic "predict skin cancer from 50 trillion images using 17 GPUs" project.
So, the next best option was doing an analysis of the best show ever - the office - inspired by the probably best data science blogpost ever from The Pudding (see [link](https://pudding.cool/2017/09/hip-hop-words/))

## Features
- Which lines / words are most characteristic of different characters (e.g., Michael: "that's what she said")?
- Which characters speak the most words per season & gender?
- Is the episode rating (if available) a function of who spoke how much?
- Are there topic clusters of different shows (bonus)

## Setup

1. Clone the repo
2. Setup the environment
3. Try stuff

### Clone repo
To clone the repo enter the following code in your terminal  
```git clone git@github.com:mattgra/the_office.git```

### Conda env
To re-create the python environment you need 2 things  
- conda (version used in this repo is `conda 4.11.0`)
- environment.yml file 

From the environment.yml file you can re-create the env via  
```conda create -f environment.yml```

### Formatting
To follow best-practices for code formatting, _black_ (version 22.3.0) was used with a line-length of 120.  
```black core/ --line-length 120```

### Testing
todo