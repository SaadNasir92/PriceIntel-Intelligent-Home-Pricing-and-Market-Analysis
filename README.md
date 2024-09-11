# Housing Price Prediction with Sentiment Analysis and Market Segmentation

## Project Overview

This project aims to develop a predictive pricing model for new housing communities using synthetic data, enhanced by natural language processing (NLP) for sentiment analysis and clustering for market segmentation. The final deliverable will be an interactive user interface for live price predictions and recommendations.

### Key Components
1. Synthetic Data Generation (Completed)
2. NLP Sentiment Analysis (Completed)
3. Predictive Pricing Model (Completed)
4. Clustering Analysis (Completed)
5. Interactive User Interface (In Development)

## Completed Components

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
- **Results**:
  - Sentiment score range: -0.4588 to 0.7424
  - Mean sentiment score: 0.073690
  - Standard deviation: 0.370224

### 3. Predictive Pricing Model
- **Status**: Completed
- **Model**: Gradient Boosting Regressor (GBR)
- **Performance Metrics**:
  - R2 Score: 0.8379
  - RMSE: 70,554.47
  - MAE: 56,170.09
- **Key Features**: Feature engineering, hyperparameter tuning, feature importance analysis
- **Best Hyperparameters**:
  - learning_rate: 0.072
  - max_depth: 2
  - max_features: 0.95
  - min_samples_leaf: 3
  - n_estimators: 99
  - subsample: 0.99

### 4. Clustering Analysis
- **Status**: Completed
- **Algorithm**: K-means clustering
- **Number of Clusters**: 5
- **Key Implementations**:
  - Feature selection using mutual information and f-regression
  - Elbow method for determining optimal cluster number
  - Cluster visualization using PCA
  - Detailed cluster analysis and characterization
- **Results**:
  - Identified 5 distinct market segments
  - Silhouette Score: 0.08478412337466534
  - Cluster sizes range from 14.81% to 23.02% of the market
  - Key differentiating factors: marital status, location preferences, customer feedback, economic indicators, and property characteristics

## In Development

### 5. Interactive User Interface
- **Objective**: Create a user-friendly interface for live price predictions and recommendations
- **Planned Features**:
  - Three-page interface:
    1. Initial Interface: Live home price predictions without buyer data
    2. Detailed Prediction and Clustering: Refined predictions with buyer data and cluster assignment
    3. Enriched Cluster Segments Review: In-depth analysis of cluster segments
  - Input form for user to enter housing and demographic data
  - Real-time price prediction display
  - Visualization of key factors influencing the prediction
  - Recommendations based on market segmentation

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

## Next Steps

1. Develop and implement the structure of the interactive user interface
2. Integrate clustering results with the predictive pricing model in the UI
3. Develop targeted marketing strategies for each identified market segment
4. Conduct thorough testing and validation of all components
5. Prepare final presentation, highlighting:
   - Predictive model performance and key price-driving features
   - Clustering analysis insights for targeted marketing strategies
   - Interactive nature of the user interface and its real-world applications

## Installation and Usage

(To be updated as development progresses)

## Contributors

(List of team members and their roles)

## License

(Specify the license under which this project is released)
