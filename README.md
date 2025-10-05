# 🎓 Student Performance Prediction System

A machine learning application that predicts student exam scores based on various factors like study hours, attendance, family background, and more.

## 📋 Project Structure

```
student-performance-ml/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Data analysis and visualization
│   └── 02_model_training.ipynb        # Model training and evaluation
│
├── data/
│   ├── StudentPerformanceFactors.csv  # Original dataset
│   └── cleaned_student_data.csv       # Cleaned dataset (generated)
│
├── models/
│   ├── best_model.pkl                 # Trained model (generated)
│   ├── scaler.pkl                     # Feature scaler (generated)
│   └── feature_names.pkl              # Feature names (generated)
│
├── app.py                             # Streamlit web application
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd student-performance-ml
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download the dataset**
   - Place `StudentPerformanceFactors.csv` in the `data/` folder
   - You can download it from [Kaggle](https://www.kaggle.com/)

## 📊 Usage

### Step 1: Data Exploration

Run the first notebook to explore and clean the data:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This will:

- Analyze the dataset
- Visualize distributions and correlations
- Clean missing values
- Generate `cleaned_student_data.csv`

### Step 2: Model Training

Run the second notebook to train and evaluate models:

```bash
jupyter notebook notebooks/02_model_training.ipynb
```

This will:

- Train Linear and Polynomial Regression models
- Compare model performance
- Save the best model and related files

### Step 3: Run Streamlit App

Launch the web application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 Features

### Streamlit App Pages:

1. **🏠 Home**

   - Dataset overview and statistics
   - Exam score distribution

2. **📊 Data Explorer**

   - Interactive visualizations
   - Feature analysis
   - Correlation heatmaps

3. **📈 Model Performance**

   - Model metrics (R², RMSE, MAE)
   - Feature importance

4. **🎯 Make Prediction**
   - Interactive prediction form
   - Real-time score prediction
   - Personalized recommendations

## 📈 Model Information

### Features Used (19 total):

- **Numerical:** Hours_Studied, Attendance, Sleep_Hours, Previous_Scores, Tutoring_Sessions, Physical_Activity
- **Categorical:** Gender, School_Type, Parental_Involvement, Access_to_Resources, Internet_Access, Motivation_Level, Family_Income, Teacher_Quality, Peer_Influence, Learning_Disabilities, Parental_Education_Level, Distance_from_Home, Extracurricular_Activities

### Target Variable:

- **Exam_Score:** Student's exam score (0-100)

### Models Trained:

1. Linear Regression (Baseline)
2. Polynomial Regression (Degree 2)
3. Polynomial Regression (Degree 3)

## 🌐 Deployment to Streamlit Cloud

1. **Push to GitHub**

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo>
git push -u origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `app.py` as the main file
   - Click "Deploy"

## 📝 Dataset Information

- **Total Records:** 6,607 students
- **Features:** 20 columns
- **Target:** Exam_Score (0-100)
- **Missing Values:**
  - Teacher_Quality: 78 (1.18%)
  - Parental_Education_Level: 90 (1.36%)
  - Distance_from_Home: 67 (1.01%)

## 🔧 Troubleshooting

**Issue:** Model files not found

- **Solution:** Run `02_model_training.ipynb` first to generate model files

**Issue:** Dataset not found

- **Solution:** Place the CSV file in the `data/` folder

**Issue:** Import errors

- **Solution:** Install all requirements: `pip install -r requirements.txt`

## 📚 Technologies Used

- **Python 3.8+**
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Machine learning
- **Streamlit** - Web app framework
- **Plotly** - Interactive visualizations
- **Matplotlib & Seaborn** - Static visualizations
- **Joblib** - Model persistence

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Your Name - [https://github.com/Rasath16]

## 🙏 Acknowledgments

- Dataset from Kaggle: Student Performance Factors
- Streamlit for the amazing framework
- Scikit-learn for machine learning tools

---

**Made with ❤️ and Python**
