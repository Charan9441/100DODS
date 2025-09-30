# 📊 Machine Learning Analysis App

A comprehensive web-based application built with Streamlit for data analysis, visualization, and machine learning model training and deployment.

## 🌟 Features

### 📂 Data Upload
- Upload custom CSV datasets
- Load pre-built demo datasets:
  - **Classification**: Iris, Wine, Breast Cancer, Synthetic Classification
  - **Regression**: California Housing
  - **Clustering**: Synthetic Blobs

### 🔍 Exploratory Data Analysis (EDA)
- Dataset overview with shape and data types
- Missing values detection
- Statistical summary
- Interactive correlation heatmap
- Data distribution visualization

### 📈 Regression
- Linear Regression modeling
- Performance metrics (MSE, R² Score)
- Predictions vs Actual values visualization
- Automatic handling of missing values

### 🔢 Classification
Multiple classification algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes
- Gradient Boosting
- Voting Classifier (Ensemble)

**Outputs:**
- Accuracy metrics
- Classification report
- Confusion matrix heatmap

### 📉 Clustering
- K-Means clustering
- Configurable number of clusters (2-10)
- Cluster distribution visualization
- 2D scatter plot of clusters

### 🚀 Model Deployment
- Save trained models in session
- Make predictions on new data:
  - Upload CSV for batch predictions
  - Manual input for single predictions
- Download predictions as CSV

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Charan9441/100DODS.git
cd 100DODS
```

2. **Create a virtual environment** (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

## 🚀 Usage

1. **Run the application**
```bash
streamlit run ml_app.py
```

2. **Access the app**
   - The app will automatically open in your default browser
   - Or navigate to `http://localhost:8501`

3. **Navigate through the app**
   - Use the sidebar to select different tasks
   - Upload your data or use demo datasets
   - Train models and make predictions

## 📦 Dependencies

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

## 📝 Workflow Example

### Classification Example

1. **Upload Data**: Go to "Data Upload" → Select "Iris" dataset
2. **Explore**: Navigate to "Exploratory Data Analysis" → View correlations
3. **Train Model**: Go to "Classification" → Select target → Choose "Random Forest" → Click "Run Classification"
4. **Deploy**: Navigate to "Model Deployment" → Enter manual input or upload CSV → Get predictions

### Regression Example

1. **Upload Data**: Load "California Housing" dataset
2. **EDA**: Check statistical summary and correlations
3. **Train**: Select target variable (e.g., "MedHouseVal") → Run Regression
4. **Evaluate**: View MSE, R² Score, and prediction plots

## 🎯 Key Features

- **User-Friendly Interface**: Intuitive navigation with sidebar menu
- **Multiple ML Algorithms**: 8+ classification algorithms to choose from
- **Visual Analytics**: Interactive plots and heatmaps
- **Model Persistence**: Keep trained models in session for predictions
- **Export Results**: Download predictions as CSV files
- **Error Handling**: Robust error messages and data validation

## 🔧 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Reinstall dependencies
pip install --upgrade streamlit pandas numpy matplotlib seaborn scikit-learn
```

**Port Already in Use**
```bash
# Use a different port
streamlit run ml_app.py --server.port 8502
```

**Missing Data Errors**
- Ensure your CSV has proper headers
- Check for missing values in critical columns
- Use numeric data for regression/clustering

## 📊 Screenshots

### Data Upload
Upload your own CSV or select from demo datasets.

### EDA Dashboard
View comprehensive statistics and correlation heatmaps.

### Model Training
Train multiple ML models with a single click.

### Predictions
Make predictions on new data and download results.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Charan**
- GitHub: [@Charan9441](https://github.com/Charan9441)
- Project: [100 Days of Data Science](https://github.com/Charan9441/100DODS)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Machine Learning powered by [scikit-learn](https://scikit-learn.org/)
- Data visualization with [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## 🐛 Bug Reports

Found a bug? Please open an issue on GitHub with:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Screenshots (if applicable)

## 💡 Future Enhancements

- [ ] Add deep learning models
- [ ] Support for time series analysis
- [ ] Model comparison dashboard
- [ ] Hyperparameter tuning interface
- [ ] Export trained models (pickle/joblib)
- [ ] Support for more file formats (Excel, JSON)
- [ ] Advanced feature engineering tools
- [ ] Model performance tracking over time

---

⭐ If you find this project helpful, please give it a star on GitHub!

**Happy Learning! 🚀**
