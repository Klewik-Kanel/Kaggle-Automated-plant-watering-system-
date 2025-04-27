# Automatic Water Supply System using Machine Learning

## 🌱 Project Description

This project builds an **automatic water supply prediction system** using **machine learning** models, based on real-time environmental sensor data. It predicts when to water plants based on soil moisture, humidity, temperature, pH, wind conditions, etc., helping **gardeners** and **farmers** automate irrigation and conserve resources effectively.

## 🚀 Problem Statement

Manual watering can lead to overwatering or underwatering, causing harm to crops. This system automates the decision based on **soil and atmospheric data**, enhancing efficiency and reducing human effort.

## 🔧 Features

- Real-time sensing and auto-decision for watering
- Complete elimination of human intervention
- Smart self-controlling relay motor activation
- Accurate predictions using KNN classifier
- Detailed data visualization and correlation analysis

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Statsmodels  
- **Other Tools:** Kaggle (dataset source)

## 📚 Dataset Details

Dataset: [Predicting Watering the Plants](https://www.kaggle.com/datasets/nelakurthisudheer/dataset-for-predicting-watering-the-plants)  
Features used:
- Soil Moisture
- Temperature
- Soil Humidity
- Time
- Air Temperature (°C)
- Wind Speed (Km/h)
- Air Humidity (%)
- Wind Gust (Km/h)
- Pressure (KPa)
- pH
- Additional Environmental Parameters

## 🧠 Model Building and Training

- **Preprocessing:**  
  - Handled missing values using **Iterative Imputer**.
  - Encoded categorical outputs using **LabelBinarizer**.
  - Scaled feature values between -1 and 1 using **MinMaxScaler**.
- **Exploratory Data Analysis (EDA):**  
  - Correlation heatmaps, distribution plots, cluster maps, and histograms were generated.
- **Model:**  
  - Trained using **K-Nearest Neighbors (KNN)** with 3 neighbors.
  - Achieved an **accuracy of ~98%** on the test set.
- **Evaluation:**  
  - Confusion matrix visualization
  - Classification report (Precision, Recall, F1-Score)

## 📈 Results

- Model accurately predicts whether watering is required based on input sensor data.
- Real-time prediction using manual input arrays is supported.

## 🖥️ How to Run

1. Clone this repository.
2. Install required libraries:
    ```bash
    pip install pandas numpy scikit-learn seaborn matplotlib statsmodels
    ```
3. Download the dataset from Kaggle and place it in the project directory.
4. Run the `notebook` or `python script` to train the model and make predictions.

## 🔮 Future Work

- Integrate model into an IoT-based hardware system for full automation.
- Experiment with other ML algorithms (Random Forest, SVM, XGBoost).
- Build a Streamlit-based web app for live monitoring.

## 👨‍💻 Author

- **Your Name** — Machine Learning Developer | Data Enthusiast

---

Would you also like me to quickly create a `requirements.txt` and example folder structure (`/data`, `/notebooks`, etc.) if you plan to make the GitHub repo even cleaner? 🚀  
Would take just a minute!
