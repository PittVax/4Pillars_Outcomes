# 4Pillars_CMI_Outcomes

==============================

Analysis of an implemenation of the 4 Pillars Practice Transformation Program for Immunization in a primary care organization.

## Project Organization

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make requirements` and `make jupyter`
    ├── README.md          <- The top-level README.
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── reports            <- Generated output as HTML, PDF, LaTeX, etc.
    │
    │── environment.yml   <- The requirements file for reproducing the analysis environment with a conda environment.
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment with a virtualenv.
                              Generated with `pip freeze > requirements.txt`

------------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Installation instructions

* Clone this repo to your local machine and navigate into the project.

```bash
git clone https://github.com/PittVax/4Pillars_Outcomes.git 4Pillars_Outcomes
cd 4Pillars_Outcomes
```

* Install requirements.

```bash
make requirements
```

* Start Jupyter notebook
```bash
make jupyter
```

## Caveats

To protect the privacy of the participating sites, the full dataset is not included with this repo. However, aggregated tables are available in `reports/tables`.
