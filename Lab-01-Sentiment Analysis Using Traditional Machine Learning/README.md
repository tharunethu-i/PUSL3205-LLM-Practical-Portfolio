# Lab 01 – Sentiment Analysis Using Traditional Machine Learning

## 1. Laboratory Overview

This practical focuses on developing a basic sentiment-analysis system using traditional machine-learning techniques. The objective is to classify movie reviews into positive and negative sentiments by applying text preprocessing, feature extraction, model training, and evaluation.

## 2. Objectives

- Understand the basic concepts of sentiment analysis.
- Load and explore a labelled text dataset.
- Perform text preprocessing.
- Identify frequently occurring words in positive and negative reviews.
- Visualise frequently occurring words.
- Convert text into numerical features using the Bag of Words technique.
- Train Logistic Regression and Support Vector Machine (SVM) models.
- Evaluate model performance using classification metrics and confusion matrices.
- Test the trained model using new sentences.

## 3. Dataset

The practical uses the `imdb_labelled.txt` dataset, which contains movie reviews with sentiment labels.

| Dataset information | Details |
|---|---:|
| Total reviews | 748 |
| Positive reviews (1) | 386 |
| Negative reviews (0) | 362 |
| Training samples | 598 |
| Testing samples | 150 |

The dataset was divided into training and testing sets using an 80:20 split with stratified sampling.

## 4. Implementation

The practical was completed using Python and the following steps:

### 4.1 Library Importation
Imported the required libraries, including Pandas, NumPy, Matplotlib, Seaborn, NLTK and Scikit-learn.

### 4.2 Dataset Loading
Loaded the IMDB labelled dataset and examined its structure, records and sentiment distribution.

### 4.3 Text Preprocessing
Applied basic text-cleaning operations:
- Converted text into lowercase.
- Removed punctuation and unnecessary characters.
- Removed extra whitespace.

### 4.4 Text Exploration
Displayed positive and negative review examples and identified the ten most frequently occurring words in each sentiment group.

### 4.5 Data Visualisation
Created bar charts to compare frequently occurring words in positive and negative reviews.

### 4.6 Dataset Splitting
Split the dataset into training and testing sets using an 80:20 ratio.

### 4.7 Feature Extraction
Applied the Bag of Words (BoW) technique using `CountVectorizer` to convert text into numerical representations. The resulting feature matrix contained 2,637 features.

### 4.8 Model Training
Trained two traditional machine-learning models:
- Logistic Regression
- Support Vector Machine (SVM) with a linear kernel

### 4.9 Model Evaluation
Evaluated both models using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### 4.10 Sentiment Prediction
Used the trained Logistic Regression model to predict the sentiment of two new movie-review sentences.

## 5. Results

### 5.1 Model Performance Comparison

| Evaluation Metric | Logistic Regression | SVM |
|---|---:|---:|
| Accuracy | 72.00% | 70.67% |
| Negative F1-score | 0.71 | 0.70 |
| Positive F1-score | 0.73 | 0.71 |
| Macro F1-score | 0.72 | 0.71 |

### 5.2 New Sentence Predictions

Two new sentences were tested using the Logistic Regression model.

| Test sentence | Predicted sentiment |
|---|---|
| The movie was absolutely brilliant with stunning visuals! | Positive |
| Horrible pacing, completely wasted my time. | Negative |

Both sentences received the expected sentiment predictions.

### 5.3 Main Finding

Logistic Regression achieved a slightly higher accuracy (72%) than SVM (70.67%). Therefore, Logistic Regression was selected for the final sentiment-prediction test.

## 6. Observations

- The dataset contains a relatively balanced distribution of positive and negative reviews.
- Text preprocessing helped standardise the reviews before feature extraction.
- Positive and negative reviews showed differences in frequently occurring words.
- Bag of Words successfully converted text into numerical features for machine-learning models.
- Both Logistic Regression and SVM successfully learned from the training data.
- Logistic Regression performed slightly better than SVM on the test dataset.
- The new sentence tests demonstrated that the trained model could predict sentiment from unseen text.
- The models may still struggle with complex language, context, sarcasm and expressions that have different meanings depending on their usage.

## 7. Learning Outcomes

Through this practical, I gained experience in:

- Loading and exploring text datasets using Pandas.
- Performing basic text preprocessing using Python and regular expressions.
- Analysing word frequencies using NLTK and Counter.
- Visualising text data using Matplotlib and Seaborn.
- Understanding and applying the Bag of Words technique.
- Training traditional machine-learning models for text classification.
- Evaluating classification models using different performance metrics.
- Applying a trained model to predict sentiment from new sentences.

## 8. Conclusion

This practical provided hands-on experience in developing a sentiment-analysis system using traditional machine-learning techniques. I learned how to prepare text data, extract numerical features, train classification models and evaluate their performance. Logistic Regression achieved a slightly better accuracy than SVM in this experiment. Overall, this practical improved my understanding of how machine learning can be applied to natural-language text classification and highlighted some limitations of traditional approaches.

## 9. Tools and Technologies

- Python
- Google Colab / Jupyter Notebook
- Pandas
- NumPy
- NLTK
- Matplotlib
- Seaborn
- Scikit-learn
