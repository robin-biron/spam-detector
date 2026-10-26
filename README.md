# Spam detector

**[Open the live application](https://robin-biron-spam-detector.streamlit.app/)** · Hosted for free on Streamlit Community Cloud.

A machine learning project for classifying spam and legitimate emails, with a research notebook and an interactive Streamlit article.

Read the related [French article on Medium](https://medium.com/datascientest/le-machine-learning-au-service-de-la-cybers%C3%A9curit%C3%A9-dans-la-d%C3%A9tection-de-spams-dans-les-e-mails-5bb072258057), published by Robin Biron in DataScientest on March 3, 2021.

## Structure

```text
spam-detector/
├── app.py                       # Streamlit tabs and interactive article
├── email_demo.py                # Text predictions and demonstration forms
├── project_paths.py             # Shared locations for data and images
├── notebooks/
│   └── spam_detector.ipynb       # Email processing, exploration and models
├── assets/
│   └── images/                  # One copy of each article illustration
├── data/
│   ├── stopwords/english.txt     # Bundled NLTK English stopwords
│   ├── processed/spam_ham.csv    # Prepared dataset used by the app
│   └── raw/                     # Local email corpus, ignored by Git
├── tests/                       # Layout, resource and corpus-path checks
└── requirements.txt             # Tested application dependency versions
```

The app and notebook use the same project paths. Images and the prepared CSV are stored once. The raw email corpus and prepared feature table have different purposes and are not duplicate datasets.

## Interactive article

Use Python 3.12 and the tested dependency versions:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The application finds its images and prepared CSV relative to the project, so it can also be launched with an absolute path to `app.py` from another directory.

The article, *Building a Spam Detector from Scratch Using Machine Learning*, presents email classification, feature engineering and model comparisons. The original notebook and article credits are retained in their sources.

The hosted app has three tabs:

- **Spam & ham examples**: analyze fictional English messages and compare predictions with their illustrative labels.
- **Analyze an email**: submit an English email body for a prediction and an estimated spam probability.
- **Methodology & results**: read the original article, both model pipelines and their reproducible evaluation results.

The testing tabs reuse Model A's fitted text-only logistic regression and vocabulary, without retraining on submitted text. They do not apply the combined models, which also require engineered numeric features and email metadata. The historical English corpus limits their relevance to modern email; model probabilities are not calibrated security risk scores. Input is limited to 20,000 characters. Submitted text is processed by the Streamlit server and is not saved by the application or sent to an external AI service.

These testing screens are a new implementation; the original interface described in the Medium publication was not included in the merged repositories.

## Research notebook

Open [notebooks/spam_detector.ipynb](notebooks/spam_detector.ipynb) in Jupyter. The notebook preserves its historical code and saved outputs. Its original dependency versions are recorded in `notebooks/requirements-legacy.txt`; it also uses Jupyter, BeautifulSoup4 and wordcloud. Reproducing the research requires a compatible historical environment, separate from the current app environment.

The original research processes raw emails from the Apache SpamAssassin public corpus. Place your extracted corpus in `data/raw/spam/` and `data/raw/ham/`, or set `SPAM_DETECTOR_RAW_DATA` to an absolute directory containing those two folders before starting Jupyter. Raw emails are not included in this repository.

The notebook now has one configuration cell for the input paths. Missing folders produce an explicit error instead of relying on a contributor's local directories. Historical computations and saved outputs are retained; reproducing them requires the original corpus and a compatible environment.

## Check the organization

The structural checks use only the Python standard library:

```sh
python -m unittest discover -s tests -v
```

They validate resource resolution from another working directory, notebook configuration, data paths, local documentation links and the absence of duplicate image files. When the application dependencies are installed, the runtime test also executes both models and checks a cached rerun. Without those dependencies, only the structural tests run.

## Deployment and reproducibility

The app uses a tested Python 3.12 environment with dependency versions pinned in `requirements.txt`. It uses native Streamlit charts, sparse text matrices and cached fitted models to run on a small hosted instance. The train/test split and estimators use a fixed seed. The CSV index is read as an index, not used as a model feature.

The app's English stopwords are bundled from the [NLTK stopwords corpus](https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/stopwords.zip), so the deployed app does not download a language corpus at runtime. The displayed live accuracy scores are recalculated from the bundled historical dataset; the original article figures and commentary describe the 2020 experiment and may differ.

For Streamlit Community Cloud, deploy this repository's `master` branch, select `app.py` and Python 3.12. No API key or paid service is required.

Local use needs only the Streamlit command above. Community Cloud automatically deploys updates pushed to `master`.

## Git history

The original notebook and interactive article histories remain in this single repository, without squashing or rewriting their commits. The former article repository has been removed. This cleanup moves files and removes the verified duplicate `Spam_img.jpeg`; its identical copy is retained as `assets/images/hacker.jpeg`.
