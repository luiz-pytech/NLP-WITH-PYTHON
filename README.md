# NLP WITH PYTHON: Challenges and Studies

![Banner NLP](https://user-images.githubusercontent.com/11280273/198124119-e33a18f4-a4f6-4993-9791-a1827b58b4f1.png)

## 📖 About The Project

This repository was created as a practical study project to consolidate the knowledge acquired in a **Natural Language Processing (NLP)** course, inspired by the DataCamp learning track. The goal is to practically apply the main techniques and tools of the NLP ecosystem in Python, from fundamental text manipulation to the implementation of advanced models with the Hugging Face `transformers` library.

Each folder corresponds to a chapter of the course and contains notebooks with challenging exercises designed to reinforce theoretical learning with real-world implementation.

---

## 🚀 Topics Covered

The journey through this repository covers the essential pillars of Natural Language Processing:

1.  **Fundamentals of Text Processing:** Text preparation and cleaning techniques, including tokenization, stopword removal, normalization (lowercasing), stemming, and lemmatization.

2.  **Feature Extraction:** Methods for transforming text into numerical representations that Machine Learning models can understand, such as **Bag-of-Words (BoW)**, **TF-IDF**, and the exploration of **Word Embeddings** (Word2Vec, GloVe).

3.  **Text Classification with Hugging Face:** Leveraging the power of pre-trained models (*transformers*) for advanced tasks such as **Sentiment Analysis**, **Zero-Shot Classification**, semantic similarity checking, and grammar correction.

4.  **Token Classification and Text Generation:** Modern applications such as **Named Entity Recognition (NER)**, Part-of-Speech (PoS) tagging, **automatic summarization**, **translation**, and **text generation**, also using Hugging Face pipelines.

---

## 📂 Repository Structure

The project is organized into directories, with each one corresponding to a study module:

/NLP_WITH_PYTHON
|
|-- 01_fundamentals_and_words_processing/
|   |-- 01_tokenization.ipynb
|   |-- 02_clearning_and_normalization.ipynb
|   -- 03_stemming_and_lemmatization.ipynb | |-- 02_feature_extraction/ |   |-- 01_bag_of_words.ipynb |   |-- 02_TF_IDF.ipynb |   -- 03_words_embeddings.ipynb
|
|-- 03_classification_with_HuggingFace/
|   |-- 01_sentimental_analysis.ipynb
|   |-- 02_zero_shot_classification.ipynb
|   -- 03_similiraty_and_grammar.ipynb | |-- 04_tokens_and_text_generation/ |   |-- 01_recognition_of_named_entity_NER.ipynb |   |-- 02_text_summary.ipynb |   |-- 03_automatic_translation.ipynb |   -- 04_simple_autocomplete_system.ipynb
|
|-- README.md
`-- requirements.txt

---

## 🛠️ Technologies and Libraries

This project was developed using **Python 3** and the following main libraries:

* **Jupyter Notebook:** For creating the exercises interactively.
* **Pandas:** For data manipulation and analysis.
* **NLTK (Natural Language Toolkit):** For fundamental tasks like tokenization and stemming.
* **spaCy:** For efficient text processing, especially for lemmatization and NER.
* **Scikit-learn:** For Bag-of-Words and TF-IDF implementations.
* **Hugging Face `transformers`:** For using state-of-the-art models in classification, generation, and other tasks.
* **Matplotlib & Seaborn:** For data visualization, such as word frequencies.

---

## 📊 Datasets

The exercises primarily use public datasets of text in Portuguese, with a focus on product reviews. An excellent data source used is the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), available on Kaggle.

---

## 🚀 How to Use

To explore this repository:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SEU-USUARIO/NLP_WITH_PYTHON.git](https://github.com/SEU-USUARIO/NLP_WITH_PYTHON.git)
    ```

2.  **Navigate to the directory:**
    ```bash
    cd NLP_WITH_PYTHON
    ```

3.  **Install the dependencies:**
    (It is recommended to create a virtual environment first)
    ```bash
    pip install -r requirements.txt
    ```

4.  **Explore the notebooks:** Open the notebooks in each folder to see the challenges and my solutions. Feel free to run the code and experiment with your own texts!

---

## ✍️ Author

* **LUIZ FELIPE**

---