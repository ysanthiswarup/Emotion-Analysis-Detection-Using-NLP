# Emotion Detection Analysis Using NLP

This project performs emotion detection from textual data using natural language processing (NLP) techniques and machine learning/deep learning models. It processes text data to classify emotions such as happiness, sadness, anger, etc.

## Features

### Data Preprocessing
- Handles text preprocessing using techniques like:
  - Removing punctuation and stopwords.
  - Lemmatizing words to their base forms.
- Libraries: `nltk`, `re`, `pandas`, `numpy`.

### Data Visualization
- Visualizes data distributions and patterns using:
  - WordCloud for text visualization.
  - Seaborn and Matplotlib for data distributions and insights.

### NLP Techniques
- Encodes text data using TF-IDF vectorization and tokenization.
- Prepares data for model training using techniques like padding and sequence encoding.

### Models Implemented
- **Deep Learning Models**:
  - LSTM (Long Short-Term Memory) networks with embedding and bidirectional layers.
- **Deep Learning Models**:
  - BERT.
  - DistilBERT.
  - RoBERTa.

### Evaluation Metrics
- Metrics used to evaluate the models:
  - Accuracy.
  - F1 Score.
  - Confusion Matrix.
  - Classification Report.

## Requirements
- Python 3.x
- Libraries:
  - `numpy`, `pandas`, `matplotlib`, `seaborn`
  - `nltk`, `tensorflow`, `scikit-learn`, `wordcloud`

## Dataset
The dataset consists of tweets annotated with emotions. The data is read from a file named `tweets.txt` and preprocessed for analysis.

## How to Run
1. Install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```
2. Load the notebook `Emotion_Analysis_Detection.ipynb` in Jupyter or Google Colab.
3. Follow the steps outlined in the notebook to preprocess the data, train the models, and evaluate their performance.

## Results
The notebook provides visualizations, training results, and evaluation metrics for each model to analyze the effectiveness of emotion detection.
