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
- [Output](#Output)
  - [What's in the data](#What's-in-the-data)
  - [Who is speaking the most per season](#Who-is-speaking-the-most-per-season)
  - [What lines define the main characters](#What-lines-define-the-main-characters)
  - [What or who makes an episode great](#What-or-who-makes-an-episode-great)
## Purpose
I was looking for a nice data science project - nice as in novel, helpful to humanity, ... - but didn't bother doing the classic "predict skin cancer from 50 trillion images using 17 GPUs" project.
So, the next best option was doing an analysis of the best show ever - the office - inspired by the probably best data science blogpost ever from The Pudding (see [link](https://pudding.cool/2017/09/hip-hop-words/))

## Features
- __Descriptive analysis__: Which characters speak the most words per season?
  - :white_check_mark: Create analysis script file
- __Basic NLP__: Which lines / words are most characteristic of different characters (e.g., Michael: "that's what she said")?
  - :white_check_mark: Create analysis file
  - :x: Create output report
- _Machine Learning (unsupervised)_: Given only the spoken text - can we identify "clusters" (e.g., characters, sentiment, ...)
  - :x: Create analysis file
  - :x: Create output report
  - :x: Create CLI interface
- _Machine Learning (causal / supervised)_: Can we predict the episode rating based on language features (e.g., who spoke who many lines)?
  - :x: Create analysis file
  - :x: Create output report
  - :x: Create CLI interface

## Setup

1. Clone the repo
2. Setup the environment
3. Try stuff

### Clone repo
To clone the repo enter the following code in your terminal  
```
git clone git@github.com:mattgra/the_office.git
```

### Conda env
To re-create the python environment you need 2 things  
- conda (version used in this repo is `conda 4.11.0`)
- environment.yml file 

From the environment.yml file you can re-create the env via  
```commandline
conda create -f environment.yml
```

### Formatting
To follow best-practices for code formatting, _black_ (version 22.3.0) was used with a line-length of 120.  
```commandline
black core/ --line-length 120
```

### Testing
To run unit tests (test data is provided in repo), run
```commandline
pytest tests
```

## Output
(TODO: this will be moved into a seperate folder)

### What's in the data
TODO

### Who is speaking the most per season
![Alt text](docs/analysis_outputs/count_of_spoken_lines_per_season_and_character.png?raw=true "The Office")

### What lines define the main characters
![Alt text](docs/analysis_outputs/tf-idf-analysis.png?raw=true "The Office")

### What or who makes an episode great
TODO