# Spam detector

A machine learning project for classifying spam and legitimate emails, with a research notebook and an interactive Streamlit article.

## Structure

```text
spam-detector/
├── app.py                       # Interactive Streamlit article
├── project_paths.py             # Shared locations for data and images
├── notebooks/
│   └── spam_detector.ipynb       # Email processing, exploration and models
├── assets/
│   └── images/                  # One copy of each article illustration
├── data/
│   ├── processed/spam_ham.csv    # Prepared dataset used by the app
│   └── raw/                     # Local email corpus, ignored by Git
├── tests/                       # Layout, resource and corpus-path checks
├── requirements.txt             # Original application dependency versions
└── Procfile                     # Optional Heroku process entry point
```

The app and notebook use the same project paths. Images and the prepared CSV are stored once. The raw email corpus and prepared feature table have different purposes and are not duplicate datasets.

## Interactive article

In an environment compatible with the original dependencies:

```sh
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m streamlit run app.py
```

The application finds its images and prepared CSV relative to the project, so it can also be launched with an absolute path to `app.py` from another directory.

The article, *Building a Spam Detector from Scratch Using Machine Learning*, presents email classification, feature engineering and model comparisons. The original notebook and article credits are retained in their sources.

## Research notebook

Open [notebooks/spam_detector.ipynb](notebooks/spam_detector.ipynb) in Jupyter. In addition to the application dependencies, the notebook uses Jupyter, BeautifulSoup4 and wordcloud; use versions compatible with the historical environment.

The original research processes raw emails from the Apache SpamAssassin public corpus. Place your extracted corpus in `data/raw/spam/` and `data/raw/ham/`, or set `SPAM_DETECTOR_RAW_DATA` to an absolute directory containing those two folders before starting Jupyter. Raw emails are not included in this repository.

The notebook now has one configuration cell for the input paths. Missing folders produce an explicit error instead of relying on a contributor's local directories. Historical computations and saved outputs are retained; reproducing them requires the original corpus and a compatible environment.

## Check the organization

The structural checks use only the Python standard library:

```sh
python -m unittest discover -s tests -v
```

They validate resource resolution from another working directory, notebook configuration, data paths, local documentation links and the absence of duplicate image files. They do not retrain the models or certify the legacy application runtime.

## Historical deployment and dependencies

Dependency versions remain the original 2020 versions. This reorganization does not modernize the scientific stack or establish that the former Heroku deployment is still available.

The root `Procfile` launches `app.py` using Heroku's `PORT`. It replaces the old `setup.sh` that wrote configuration into the user's home directory. Local use needs only the Streamlit command above; no deployment is performed by reorganizing this repository.

## Git history

The original `spam-detector` notebook and `article-spam-detector-heroku` histories remain in this single repository, without squashing or rewriting their commits. The former article repository has been removed. This cleanup moves files and removes the verified duplicate `Spam_img.jpeg`; its identical copy is retained as `assets/images/hacker.jpeg`.
