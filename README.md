# Housing Price Prediction with Sentiment Analysis

## Project Overview

This project aims to develop a predictive pricing model for new housing communities using synthetic data, enhanced by natural language processing (NLP) for sentiment analysis and clustering for market segmentation. The final deliverable will be an interactive user interface for live price predictions and recommendations.

### Key Components
1. Synthetic Data Generation
2. NLP Sentiment Analysis
3. Predictive Pricing Model
4.  Clustering Analysis (in development)
5. Interactive User Interface (in development)

## Current Progress

### 1. Synthetic Data Generation
- **Status**: Completed
- **Tool Used**: CTGANSynthesizer
- **Dataset Size**: 250,000 rows
- **Validation**: Initial validation confirms adherence to predefined rules and distributions

### 2. NLP Sentiment Analysis
- **Status**: Completed
- **Tool Used**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Key Implementations**:
  - Custom lexicon for domain-specific terms
  - Refined scoring mechanism for mixed sentiments
  - Integration of sentiment scores into the main dataset

#### Sample Code: Custom Lexicon Addition

```python
custom_lexicon = {
    'upgrade': 2.0, 'spacious': 1.5, 'delay': -1.5, 'issue': -1.0,
    'smooth': 1.0, 'fantastic': 2.0, 'terrible': -2.0, 'happy': 1.5, 'love': 2.0
}
sia.lexicon.update(custom_lexicon)
```

#### Sample Code: Mixed Sentiment Adjustment

```python
def adjust_mixed_sentiment(text, score):
    if "but" in text.lower() or "however" in text.lower():
        return score * 0.75 if score > 0 else score * 1.25
    return score
```

#### Results
- Sentiment score range: -0.4588 to 0.7424
- Mean sentiment score: 0.073690
- Standard deviation: 0.370224

![Sentiment Score Distribution](sentiment_distribution.png)

*Note: Placeholder, will be updated*

## Feature List

### Demographic Features
- City, Zipcode, Marital Status, Family Size, Credit Score, Current Residence, Annual Income, DTI, Location

### Structural Features
- Square Footage, Upgrade Score, Bedroom Count, Bathroom Count, Master Location, Lot Size

### Loan-Related Features
- Closing Date, Appraised Value, Price, Closing Interest Rate, Monthly Payment, Loan Term, Feedback

### Economic Features
- Inflation Rate, Unemployment Rate, Current Market Interest Rate, Year Closed, Median/Mean Price of Nearby Homes

### Sentiment Feature
- Sentiment score derived from feedback (added by NLP module)

## In Development

### 3. Clustering Analysis
- Objective: Perform market segmentation based on demographic and housing preference data
- Planned Approach: Utilize K-means or DBSCAN algorithms
- Integration: Incorporate sentiment scores from NLP analysis

### 4. Predictive Pricing Model
Model: Gradient Boosting Regressor (GBR)
Performance Metrics:
R2 Score: 0.8379
RMSE: 70,554.47
MAE: 56,170.09

Key Features: Feature engineering, hyperparameter tuning, feature importance analysis
Best Hyperparameters:
learning_rate: 0.072
max_depth: 2
max_features: 0.95
min_samples_leaf: 3
n_estimators: 99
subsample: 0.99

### 5. Interactive User Interface
- Objective: Create a user-friendly interface for live price predictions and recommendations
- Planned Features:
  - Input form for user to enter housing and demographic data
  - Real-time price prediction display
  - Visualization of key factors influencing the prediction
  - Recommendations based on market segmentation

## Next Steps

1. Begin clustering analysis, incorporating sentiment scores
3. Design and implement the structure of the interactive user interface
4. Conduct thorough testing and validation of all components
5. Prepare final presentation, highlighting the potential of clustering analysis for targeted marketing strategies

## Installation and Usage

(To be updated as development progresses)



