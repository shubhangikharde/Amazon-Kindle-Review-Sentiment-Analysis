# Amazon Kindle Review Sentiment Analysis

## 1. Project Overview

This project focuses on sentiment analysis of Amazon Kindle Store customer reviews using Natural Language Processing (NLP) and Machine Learning techniques.

The project analyzes customer review text and classifies reviews as positive or negative based on their ratings. It uses text preprocessing, Bag of Words (BOW), TF-IDF vectorization, and the Gaussian Naive Bayes algorithm to train and evaluate sentiment classification models.

## 2. Objectives

- Analyze customer reviews from the Amazon Kindle Store.
- Clean and preprocess textual data.
- Handle missing values and prepare review data.
- Convert text into numerical features using BOW and TF-IDF.
- Classify customer reviews as positive or negative.
- Train a Gaussian Naive Bayes classification model.
- Evaluate model performance using accuracy and confusion matrices.

## 3. Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas
- **Natural Language Processing:** NLTK
- **Text Cleaning:** Regular Expressions (re)
- **HTML Processing:** BeautifulSoup
- **Feature Extraction:** CountVectorizer and TfidfVectorizer
- **Machine Learning:** Gaussian Naive Bayes
- **Model Evaluation:** Scikit-learn
- **Dataset Format:** CSV

## 4. Dataset Description

The project uses the `all_kindle_review.csv` dataset, containing Amazon Kindle Store customer reviews.

The original Amazon Kindle review dataset described in the project context contains approximately 982,619 records.

### Important Columns

- `reviewText` – Text written by the customer.
- `rating` – Customer rating for the product.
- `asin` – Product identifier.
- `reviewerID` – Unique reviewer identifier.
- `reviewerName` – Name of the reviewer.
- `summary` – Short summary of the review.
- `reviewTime` – Date of the review.
- `unixReviewTime` – Unix timestamp of the review.
- `helpful` – Helpfulness information for the review.

For this project, the analysis focuses on the `reviewText` and `rating` columns.

## 5. Data Preprocessing and Cleaning

The following preprocessing steps are performed:

1. Loaded the dataset using Pandas.
2. Selected the `reviewText` and `rating` columns.
3. Checked the dataset shape and missing values.
4. Examined the unique ratings and their distribution.
5. Converted ratings into binary sentiment labels:
   - Rating below 3 → Negative (0)
   - Rating of 3 or above → Positive (1)
6. Converted review text to lowercase.
7. Removed special characters using regular expressions.
8. Removed English stopwords using NLTK.
9. Removed URLs from review text.
10. Removed HTML tags using BeautifulSoup.
11. Removed extra spaces from the text.
12. Applied lemmatization using WordNetLemmatizer.

These steps help prepare the review text for machine learning.

## 6. Train-Test Split

The dataset is divided into training and testing sets using `train_test_split`.

- **Training Set:** 80%
- **Testing Set:** 20%

The training set is used to train the machine learning models, while the testing set is used to evaluate their performance.

## 7. Feature Extraction

Machine learning algorithms require numerical input. Therefore, two text vectorization techniques are used.

### 7.1 Bag of Words (BOW)

Bag of Words converts review text into numerical feature vectors based on word frequency.

In this project, `CountVectorizer` is used to transform the text into numerical representations.

### 7.2 TF-IDF

TF-IDF stands for Term Frequency–Inverse Document Frequency.

It assigns importance to words based on their frequency in a review and how common or rare they are across the document collection.

In this project, `TfidfVectorizer` is used to create TF-IDF feature vectors.

## 8. Machine Learning Algorithm

### Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

Two models are trained:

1. Gaussian Naive Bayes using Bag of Words features.
2. Gaussian Naive Bayes using TF-IDF features.

Both models are evaluated using the test dataset.

## 9. Model Evaluation

The project uses the following evaluation techniques:

### Accuracy Score
Measures the proportion of correctly classified reviews.

### Confusion Matrix
Shows the number of correct and incorrect predictions for each sentiment class.

### Classification Report
Provides precision, recall, F1-score, and support for each class.

The results from the BOW and TF-IDF models can be compared to understand their classification performance.

## 10. Project Workflow

1. Import required libraries.
2. Load the Amazon Kindle review dataset.
3. Select the required columns.
4. Inspect and clean the data.
5. Convert ratings into sentiment labels.
6. Preprocess the review text.
7. Split the data into training and testing sets.
8. Apply BOW vectorization.
9. Apply TF-IDF vectorization.
10. Train Gaussian Naive Bayes models.
11. Generate predictions.
12. Evaluate and compare model performance.

## 11. How to Run the Project

### Step 1: Install Required Libraries

Run the following command in the terminal:

```bash
pip install pandas numpy nltk beautifulsoup4 lxml scikit-learn
```

### Step 2: Download NLTK Resources

Run the following code before preprocessing the reviews:

```python
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Step 3: Place the Dataset

Keep `all_kindle_review.csv` in the same folder as your Jupyter Notebook or Python script.

### Step 4: Execute the Code

Run the notebook cells in the correct order, from loading the dataset through model evaluation.

### Step 5: Review the Results

Check the confusion matrices and accuracy scores for the BOW and TF-IDF models.

## 12. Key Learning Outcomes

- Learned how to preprocess text data using Python.
- Understood the basics of Natural Language Processing.
- Practiced stopword removal and lemmatization.
- Learned how to convert text into numerical features.
- Understood Bag of Words and TF-IDF vectorization.
- Implemented Gaussian Naive Bayes classification.
- Learned how to split data into training and testing sets.
- Evaluated classification models using accuracy and confusion matrices.

## 13. Future Enhancements

- Compare additional machine learning algorithms such as Logistic Regression, Multinomial Naive Bayes, and Support Vector Machines.
- Use precision, recall, F1-score, and ROC-AUC for more comprehensive evaluation.
- Improve text preprocessing and feature selection.
- Apply cross-validation and hyperparameter tuning.
- Explore deep learning approaches for sentiment classification.
- Develop an interactive application for analyzing customer reviews.

## 14. Conclusion

The Amazon Kindle Review Sentiment Analysis project demonstrates how Natural Language Processing and Machine Learning can be used to classify customer reviews into positive and negative sentiment categories.

By applying text preprocessing, BOW, TF-IDF, and Gaussian Naive Bayes, the project provides practical experience in text classification and model evaluation.

This project serves as a foundation for more advanced sentiment analysis applications and customer feedback analysis.

## 15. Author

**Name:** Shubhangi Kharde
**Project Type:** Natural Language Processing and Machine Learning
