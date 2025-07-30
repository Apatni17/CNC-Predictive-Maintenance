# 🔧 CNC Predictive Maintenance Model Comparison

A machine learning application for comparing different models in predictive maintenance for CNC machines.

## 🎯 Features

- **Custom Data Sampling**: Choose training and testing distributions
- **Model Comparison**: Compare Decision Tree, Random Forest, Logistic Regression, Gaussian Naive Bayes, K-Nearest Neighbors, and XGBoost
- **Confusion Matrix Analysis**: Detailed analysis with practical implications
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score comparison

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd CNC-Predictive-Maintenance
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run main.py
   ```

4. **Open in browser**
   - Navigate to `http://localhost:8501`

## 🌐 Deployment Options

### Option 1: Deploy Streamlit App (Recommended)

#### Deploy to Render (Free)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - Sign up/Login with GitHub
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `cnc-predictive-maintenance`
     - **Environment**: `Python`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `streamlit run main.py --server.port $PORT --server.address 0.0.0.0`
   - Click "Create Web Service"

3. **Your app will be live at**: `https://your-app-name.onrender.com`

#### Deploy to Streamlit Cloud (Alternative)

1. **Push to GitHub** (same as above)

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and main.py
   - Click "Deploy"

### Option 2: Convert to React + Backend (Advanced)

If you want a more scalable solution with React frontend:

#### Backend (Python Flask)

```python
# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

app = Flask(__name__)
CORS(app)

@app.route('/api/compare-models', methods=['POST'])
def compare_models():
    data = request.json
    # Your model comparison logic here
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
```

#### Frontend (React)

```jsx
// App.js
import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    setFile(file);
  };

  const compareModels = async () => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await axios.post('/api/compare-models', formData);
    setResults(response.data);
  };

  return (
    <div>
      <h1>CNC Predictive Maintenance</h1>
      <input type="file" onChange={handleFileUpload} />
      <button onClick={compareModels}>Compare Models</button>
      {results && <ResultsDisplay results={results} />}
    </div>
  );
}
```

## 📁 Project Structure

```
CNC-Predictive-Maintenance/
├── main.py              # Main Streamlit application
├── utils.py             # Utility functions (if needed)
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment config
├── README.md           # This file
└── .gitignore          # Git ignore file
```

## 🔧 Configuration

### Environment Variables

For Render deployment, you can set these in the dashboard:
- `PYTHON_VERSION`: Python version (default: 3.9.16)

### Customization

You can modify:
- **Model parameters** in `main.py`
- **Sampling strategies** in the UI
- **Visualization styles** in the display functions

## 📊 Usage

1. **Upload Dataset**: CSV file with your CNC machine data
2. **Set Sampling**: Choose training/testing distributions
3. **Select Models**: Pick which models to compare
4. **View Results**: Analyze confusion matrices and metrics

## 🛠️ Development

### Adding New Models

1. Import the model in `main.py`
2. Add to `model_map` dictionary
3. Add to model selection options

### Adding New Metrics

1. Import the metric from sklearn.metrics
2. Add calculation in `compare_models()` function
3. Update display in `display_results()` function

## 📈 Performance Tips

- **Large datasets**: Consider reducing sample sizes for faster processing
- **Model selection**: Start with fewer models for quicker comparison
- **Memory usage**: Monitor memory usage with large datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the code comments

---

**Happy Deploying! 🚀**
