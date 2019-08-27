4Pillars_CMI_Outcomes
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
# Installation instructions

```
conda install -c conda-forge -y \
    jupyterlab \
    nbstripout \
    nodejs \
    ipykernel \
    nb_conda \
    nb_conda_kernels


for file in $(find /root/projects -name environment.yml); do
eval $(dos2unix $file)
eval $(parse_yaml $file)
source activate base
conda env create --file $file -n $name
conda env update --file $file -n $name
# install environment kernel in jupyter
source activate $name
conda install -c conda-forge ipykernel
ipython kernel install --name=$name
done
```

# Specify base image
FROM continuumio/miniconda3:latest

# Define environment variables in the container
ENV PROJECT_DIR=/root/projects \
    NOTEBOOK_PORT=8888 \
    SSL_CERT_PEM=/root/.jupyter/jupyter.pem \
    SSL_CERT_KEY=/root/.jupyter/jupyter.key \
    PW_HASH="u'sha1:31cb67870a35:1a2321318481f00b0efdf3d1f71af523d3ffc505'" \
    CONFIG_PATH=/root/.jupyter/jupyter_notebook_config.py \
    SHELL=/bin/bash \
    JUPYTERLAB_SETTINGS_DIR=/root/user-settings

# Add build scripts
ADD etc/ /opt/etc/
WORKDIR /opt/etc/

# Install packages
RUN apt-get update && apt-get install -y apt-utils dos2unix gcc nano\
    # Python packages from conda
    && conda install -c conda-forge -y \
    jupyterlab \
    # nbstripout \
    nodejs \
    ipykernel \
    nb_conda \
    nb_conda_kernels \
    # Create a home for the mounted volumes
    && mkdir /root/user-settings \
    && mkdir /root/credentials \
    # execute dos2unix in case the script was molested by windows
    && find . -type f -print0 | xargs -0 dos2unix \
    # make startup script executable
    && chmod +x /opt/etc/docker_cmd.sh

# Expose port 8888 to host
EXPOSE 8888

# Run additional installation steps in the container
WORKDIR ${PROJECT_DIR}
CMD ["/bin/bash", "/opt/etc/docker_cmd.sh"]

# generate configuration, cert, and password if this is the first run
if [ ! -f /var/tmp/pv-conda_init ] ; then
    jupyter notebook --allow-root --generate-config
    if [ ! -f ${SSL_CERT_PEM} ] ; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -subj "/C=US/ST=Denial/L=Springfield/O=Dis/CN=127.0.0.1" \
            -keyout ${SSL_CERT_KEY} -out ${SSL_CERT_PEM}
    fi
    echo "c.NotebookApp.password = ${PW_HASH}" >> ${CONFIG_PATH}

    # # add notebook output strip filter
    # echo "
    # def scrub_output_pre_save(model, **kwargs):
    # """scrub output before saving notebooks"""
    #     # only run on notebooks
    #     if model['type'] != 'notebook':
    #         return
    #     # only run on nbformat v4
    #     if model['content']['nbformat'] != 4:
    #         return

    #     for cell in model['content']['cells']:
    #         if cell['cell_type'] != 'code':
    #             continue
    #         cell['outputs'] = []
    #         cell['execution_count'] = None
    # c.FileContentsManager.pre_save_hook = scrub_output_pre_save
    
    # import io
    # import os
    # from notebook.utils import to_api_path

    # _script_exporter = None

    # def script_post_save(model, os_path, contents_manager, **kwargs):
    #     """convert notebooks to Python script after save with nbconvert

    #     replaces `jupyter notebook --script`
    #     """
    #     from nbconvert.exporters.script import ScriptExporter

    #     if model['type'] != 'notebook':
    #         return

    #     global _script_exporter

    #     if _script_exporter is None:
    #         _script_exporter = ScriptExporter(parent=contents_manager)

    #     log = contents_manager.log

    #     base, ext = os.path.splitext(os_path)
    #     script, resources = _script_exporter.from_filename(os_path)
    #     script_fname = base + resources.get('output_extension', '.txt')
    #     log.info("Saving script /%s", to_api_path(script_fname, contents_manager.root_dir))

    #     with io.open(script_fname, 'w', encoding='utf-8') as f:
    #         f.write(script)

    # c.FileContentsManager.post_save_hook = script_post_save
    # " >> ${CONFIG_PATH}

    # import os
    # from subprocess import check_call
    # def post_save(model, os_path, contents_manager):
    #     if model['type'] != 'notebook':
    #         return # only do this for notebooks
    #     d, fname = os.path.split(os_path)
    #     check_call(['ipython', 'nbconvert', '--to', 'script', fname], cwd=d)
    # c.FileContentsManager.post_save_hook = post_save

# create environemts for projects
    for file in $(find /root/projects -name environment.yml); do
    eval $(dos2unix $file)
    eval $(parse_yaml $file)
    source activate base
    conda env create --file $file -n $name
    conda env update --file $file -n $name
    # install environment kernel in jupyter
    source activate $name
    conda install -c conda-forge ipykernel
    ipython kernel install --name=$name
    done

    # record the first run
    touch /var/tmp/pv-conda_init
fi

jupyter lab --allow-root -y --no-browser --notebook-dir=${PROJECT_DIR} \
    --certfile=${SSL_CERT_PEM} --keyfile=${SSL_CERT_KEY} --ip='*' \
    --config=${CONFIG_PATH}

