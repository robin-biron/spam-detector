# Spam detector

A machine learning project for classifying spam and legitimate emails. This repository brings together the Python notebook and the interactive Streamlit article, including its images and prepared dataset.

## Project structure

| Component | Location |
| --- | --- |
| Python / machine learning notebook | [machine-learning/spam-detector/](machine-learning/spam-detector/) |
| Interactive Streamlit article | [article/article-spam-detector.py](article/article-spam-detector.py) |
| Article overview | [article/README.md](article/README.md) |
| Article images and prepared dataset | [article/](article/) |
| Historical dependencies and Heroku configuration | `article/requirements.txt`, `article/Procfile`, `article/setup.sh` |

## Explore the notebook

Open [the Jupyter notebook](machine-learning/spam-detector/spam-detector-jupyter-notebook.ipynb) to explore the original email processing and machine learning work. Its input email directories must be configured for your local dataset.

## Run the interactive article

The article uses relative paths for its images and CSV, so run it from the `article` directory:

```sh
cd article
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m streamlit run article-spam-detector.py
```

The dependency versions are the original 2020 versions and require a compatible Python environment. This consolidation does not update them or establish that the historical Heroku deployment is still available. The original deployment files are kept under `article/`; deployment from the repository root would require configuring that subdirectory as the application root.

## Repository history

The former [article-spam-detector-heroku](https://github.com/robin-biron/article-spam-detector-heroku) repository was imported under `article/` without squashing its commits. Both original Git histories are retained, and the notebook, application, images and dataset are preserved unchanged. `spam-detector` is the combined repository; the original article repository remains available for existing links.
