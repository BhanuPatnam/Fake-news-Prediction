# 📰 Fake News Detection - Machine Learning Project

## 🔍 Project Overview
This machine learning project focuses on detecting fake news articles using natural language processing (NLP) and logistic regression classification. The system analyzes news content to predict whether an article is **real (0)** or **fake (1)** with high accuracy.

## 📊 Dataset Information
- **Source**: CSV file containing news articles with labels
- **Size**: 20,822 records with 5 features
- **Label Encoding**:
  - `1` = Fake News
  - `0` = Real News
- **Features**: id, title, author, text, label
- Download the **Dataset:** [Fake News Dataset](https://drive.google.com/file/d/12iw6qatKrygwviBePvWzpllJJmBru_Yg/view?usp=sharing)

## 🛠️ Technologies Used
- **Python 3**
- **Libraries**:
  - pandas, numpy, matplotlib, seaborn
  - scikit-learn (TfidfVectorizer, LogisticRegression, train_test_split, accuracy_score)
  - NLTK (stopwords, PorterStemmer)
  - re (regular expressions)

## 📈 Data Processing Pipeline

### 1. **Data Preprocessing**
- Handled missing values by replacing with empty strings
- Created a combined feature: `content = author + title`
- Separated features (X) and labels (Y)

### 2. **Text Processing**
- **Stemming**: Used PorterStemmer to reduce words to their root forms
- **Stopword Removal**: Eliminated common English stopwords
- **Cleaning**: Removed non-alphabetic characters and converted to lowercase

### 3. **Feature Engineering**
- **TF-IDF Vectorization**: Converted text data to numerical features
- Result: 20,822 documents → 17,149 unique features

### 4. **Model Training**
- **Algorithm**: Logistic Regression
- **Train-Test Split**: 80-20 split with stratification
- **Pre-processing**: Handled non-numeric labels and invalid indices

## 🎯 Model Performance
- **Training Accuracy**: **98.69%**
- **Testing Accuracy**: **97.84%**
- The model demonstrates excellent performance in distinguishing between real and fake news

## 🚀 Key Features
- **High Accuracy**: Achieves nearly 98% accuracy on test data
- **Robust Preprocessing**: Comprehensive text cleaning and feature engineering
- **Stratified Sampling**: Maintains class distribution during train-test split
- **Production-Ready**: Includes a predictive system for new articles

## 📁 Project Structure

**Fake_news_prediction.ipynb** (Main Jupyter Notebook)
* ├── **Data Loading & Exploration**: Initial data analysis
* ├── **Data Preprocessing**: Handling missing values and cleaning
* ├── **Feature Engineering**: Stemming and TF-IDF Vectorization
* ├── **Model Training**: Logistic Regression implementation
* ├── **Model Evaluation**: Accuracy score and performance metrics
* └── **Predictive System**: Function to test custom news inputs

## 💡 How It Works
1. Input news article (title + author)
2. Preprocess text (stemming, stopword removal)
3. Convert to TF-IDF features
4. Predict using trained logistic regression model
5. Output: "Real News" or "Fake News"

## 🔮 Potential Applications
- **Social Media Platforms**: Automatically flag suspicious content
- **News Aggregators**: Verify article credibility
- **Educational Tools**: Teach media literacy
- **Research**: Analyze fake news patterns and characteristics

## ⚙️ Setup & Usage
```python
# Clone the repository
git clone <repository-url>

# Install required packages
pip install pandas numpy scikit-learn nltk matplotlib seaborn

# Run the Jupyter notebook
jupyter notebook Fake_news_prediction.ipynb 
```

## 💡 How It Works
1. Input news article (title + author)
2. Preprocess text (stemming, stopword removal)
3. Convert to TF-IDF features
4. Predict using trained logistic regression model
5. Output: "Real News" or "Fake News"

## 🔮 Potential Applications
- **Social Media Platforms**: Automatically flag suspicious content
- **News Aggregators**: Verify article credibility
- **Educational Tools**: Teach media literacy
- **Research**: Analyze fake news patterns and characteristics

## ⚙️ Setup & Usage
```python
# Clone the repository
git clone <repository-url>

# Install required packages
pip install pandas numpy scikit-learn nltk matplotlib seaborn

# Run the Jupyter notebook
jupyter notebook Fake_news_prediction.ipynb
```

## 📋 Requirements
* **Python 3.x**
* **Jupyter Notebook**
* **NLTK data** (stopwords)

## 🎓 Insights & Improvements
The model shows slight overfitting (**train accuracy > test accuracy**).

**Potential improvements:**
* Try other algorithms (**Random Forest, SVM, Neural Networks**)
* Hyperparameter tuning
* Cross-validation
* Additional features (source credibility, writing style metrics)
* Handle class imbalance if present

## 🤝 Contributing
Feel free to fork this repository and submit pull requests with improvements or additional features.

## 📄 License
This project is open-source and available for educational and research purposes.

> **Note:** This model is trained on a specific dataset and may need retraining for different domains or languages. Always verify critical information through multiple reliable sources.

---
**Accuracy:** 97.84% | **Algorithm:** Logistic Regression | **Features:** TF-IDF from text content
