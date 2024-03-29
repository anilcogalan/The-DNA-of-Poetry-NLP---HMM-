# The-DNA-of-Poetry (NLP-HMM)
-----------
![image](https://github.com/anilcogalan/The-DNA-of-Poetry-NLP---HMM-/assets/61653147/0d4301d3-ad68-49cf-b849-8e4eed1bd79e)

----------

This project is a Natural Language Processing (NLP) application developed to classify Turkish poems according to two different poets who are **Can Yücel** and **Nazım Hikmet**. The project processes the text of poems and predicts which category each poem belongs to using HMM techniques. The poem data was collected using Selenium, showcasing the capability to scrape complex web data for NLP tasks.


## Installation

Follow the steps below to run the project on your local machine:

* Clone the project from GitHub:

```git clone https://github.com/anilcogalan/The-DNA-of-Poetry-NLP---HMM-.git```


**Note:** You might need to download the stopwords dataset for the nltk library:

``` python -m nltk.downloader stopwords ``` 

## Dataset

The project uses Turkish poetry data stored in two different JSON files, collected via Selenium web scraping. These files are:
- `poems_cy.json`: Poems belonging to the first category
- `poems_nh.json`: Poems belonging to the second category

Each file consists of an array of objects containing the texts and links of the poems.


## Features

* **Data Cleaning:** Removing unnecessary characters and links from poem texts.
* **Text Preprocessing:** Removing Turkish stopwords, converting to lowercase, and stemming.
* **Text Vectorization:** Converting poem texts into numerical representations.
* **Model Training:** Training the model on the training dataset.
* **Prediction and Evaluation:** Evaluating the model's performance on both training and test datasets.

## Used Libraries and Technologies
- **Pandas**: For data manipulation and analysis.
- **Selenium**: For automating web browsers to scrape poem data.
- **NLTK**: For natural language processing tasks such as text preprocessing.
- **TurkishStemmer**: For stemming Turkish texts to their root forms.
- **Unidecode**: For ASCII transliterations of Unicode text.
- **Streamlit**: For turning data scripts into shareable web apps.

## Usage

To run the project, execute the main Python file from the terminal as follows:

1. Load the Needed Libraries:
   
```pip install -r .\requirements.txt``` 

----------
2.  Directly, To see the **accuracy scores both train and test:**
   
```python text_preprocessing.py```

----------
3. To use and test the algorithm:
   
```streamlit run main.py```

----------

## Contributing
Those who wish to contribute can send pull requests or open issues to solve existing problems.

## License
This project is licensed under the MIT License. For more information, see the LICENSE file.

