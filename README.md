# IMDb Movie Reviews Sentiment Analysis (NLP)

## Project Overview

Film Junky Union, a community focused on the analysis of classic movies, aims to develop an automated system to classify movie reviews based on their sentiment polarity.

The goal of this project is to train a **Natural Language Processing (NLP)** model capable of identifying **negative movie reviews** from text data, using a labeled IMDb reviews dataset with positive and negative sentiment annotations.

Model performance is evaluated using the **F1-score**, which must reach a minimum value of **0.85**, according to the project requirements.

---

## Dataset

The dataset used in this project (`imdb_reviews.tsv`) is **not included in this repository**.

This dataset was provided as part of the **TripleTen Data Science program** execution environment and was accessed internally using the following path:

/datasets/imdb_reviews.tsv

Due to licensing and platform restrictions, the dataset cannot be redistributed.

To run this project locally, you must provide an equivalent IMDb reviews dataset and update the file path in the notebook accordingly.

---

## Exploratory Data Analysis (EDA)

An exploratory analysis was conducted to better understand the structure and behavior of the data, including:

- Evolution of the number of movies and reviews over time  
- Distribution of reviews per movie  
- Rating distributions in training and test sets  
- Temporal distribution of positive and negative reviews  
- Comparison between training and test subsets  

Written observations accompany each visualization to support the modeling decisions.

---

## Models Trained

The following models were implemented and evaluated:

- **Baseline model** (Dummy Classifier)
- **Logistic Regression + TF-IDF (NLTK)**
- **Logistic Regression + TF-IDF (spaCy)**
- **LightGBM Classifier + TF-IDF (spaCy)**
- **Logistic Regression on BERT embeddings**

All classical models were evaluated using a unified evaluation routine that reports **Accuracy, F1-score, ROC AUC, and Average Precision**.

---

## Note on BERT Usage

An additional model based on **BERT embeddings** is included as an advanced NLP approach.

Due to computational limitations when running BERT on CPU, the model was trained and evaluated using only **200 training samples and 200 test samples**, following the course recommendations.

This limited sample size leads to **overfitting**, which is explicitly analyzed and discussed in the conclusions section.  
Despite this limitation, the experiment demonstrates the potential of BERT-based representations for sentiment analysis.

---

## Generated Files

During execution, BERT embeddings may be saved locally as compressed `.npz` files to avoid recomputing them.

These files are **not included in the repository** and are intentionally excluded via `.gitignore`, as they are generated artifacts and can be recreated by running the notebook.

---

## Technologies Used

- Python  
- pandas  
- NumPy  
- scikit-learn  
- NLTK  
- spaCy  
- LightGBM  
- Transformers (BERT)  
- PyTorch  
- Matplotlib / Seaborn  
- Jupyter Notebook  

---

## Conclusions

- **Logistic Regression + TF-IDF (NLTK and spaCy)** achieved excellent and stable performance, meeting the project requirements and showing strong generalization.
- **LightGBM** performed well but did not outperform linear models, likely due to the sparse nature of TF-IDF features.
- **BERT-based modeling** demonstrated clear overfitting due to the limited sample size, but still showed meaningful predictive behavior, highlighting its potential with sufficient data.
- Overall, classical NLP approaches remain highly competitive and efficient for sentiment analysis tasks on structured text data.

---

## Notes

This project was developed as part of an academic machine learning sprint in the **TripleTen Data Science program**, following professional best practices in data preprocessing, model evaluation, and result interpretation.
