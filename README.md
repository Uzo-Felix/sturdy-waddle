# Heart Disease Classification Project

## Overview
A comprehensive comparison of traditional machine learning models and Neural Formal Concept Analysis (Neural FCA) for heart disease classification. This project analyzes the UCI Heart Disease dataset using various approaches to demonstrate the trade-offs between model accuracy and interpretability.

## Dataset
- **Source**: UCI Heart Disease Dataset
- **Features**: 13 clinical parameters including age, sex, chest pain type, blood pressure
- **Target**: Binary classification (presence/absence of heart disease)
- **Size**: 297 samples

## Project Structure

heart_disease_analysis/
- ├── src/
- │   ├── main.py
- │   ├── data_loader.py
- │   ├── binarization.py
- │   ├── models.py
- │   ├── neural_lib.py
- │   ├── presentation_visuals.py
- ├── results/
- │   └── model_performance.txt
- ├── figures/
- │   ├── feature_distributions.png
- │   ├── binarization_process.png
- │   ├── model_architectures.png
- │   └── performance_heatmap.png


## Models Implemented
### Traditional Models:
- Random Forest
- XGBoost
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Naive Bayes
- CatBoost

### Neural FCA:
- Custom implementation based on concept lattices
- Interpretable architecture
- Concept-based decision making

## Key Features
- Data preprocessing and standardization
- Feature binarization for Neural FCA
- Comprehensive model comparison
- Visualization tools for analysis
- Performance metrics evaluation

## Installation

```bash
git clone [repository-url]
cd heart_disease_analysis
pip install -r requirements.txt
```


## Usage
```bash
python src/main.py
```


## Results
- Best performing model: Naive Bayes (91.67% accuracy)
- Strong performers: Random Forest (88.33%), Logistic Regression (86.67%)
- Neural FCA provides interpretable results with 60.00% accuracy

## Visualization Tools
- Feature distribution plots
- Binarization process visualization
- Model architecture diagrams
- Performance comparison heatmaps

## Dependencies
- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- networkx
- fcapy
- torch
- xgboost
- catboost

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License
MIT License

## Authors
- [ONYEKWELU UZOCHUKWU FELIX](https://github.com/uzo-felix)

## Acknowledgments
- UCI Machine Learning Repository
- Neural FCA paper authors
- Course instructors and contributors