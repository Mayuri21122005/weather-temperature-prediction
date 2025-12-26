# 🌡️ Weather Temperature Prediction using Machine Learning

This project is a **machine learning–based weather temperature prediction system** developed using **Python**.  
The application predicts **temperature** based on environmental parameters such as **humidity** and **wind speed** using a **Linear Regression** model.

This project was developed as a **learning-oriented ML project** to understand supervised learning, data handling, and model training.

---

## 📌 Project Overview

- Uses **historical weather data** stored in a CSV file  
- Applies **Linear Regression** (Supervised Learning)  
- Predicts **temperature (°C)** based on:
  - Humidity (%)
  - Wind Speed (m/s)

---

## 🧠 Machine Learning Concepts Used

- Supervised Learning  
- Regression  
- Linear Regression Algorithm  
- Training & Prediction  
- Feature Selection  

---

## 🛠️ Tech Stack

- **Python**
- **Pandas** – data handling
- **Scikit-learn** – machine learning model
- **VS Code** – development environment

---

## 📂 Project Structure

weather-temperature-prediction/
│
├── weather_ml.py
├── weather_data.csv
├── README.md


---

## 📊 Dataset Description

The dataset (`weather_data.csv`) contains the following columns:

| Column Name | Description |
|-----------|------------|
| humidity  | Humidity percentage |
| wind      | Wind speed (m/s) |
| temp      | Temperature (°C) |

Sample data:
csv 
humidity,wind,temp
30,1.5,34
45,2.5,31
60,4.0,28
75,5.5,25
How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/Mayuri21122005/weather-temperature-prediction.git
cd weather-temperature-prediction

2️⃣Install Required Libraries
pip install pandas scikit-learn

3️⃣ Run the Program
python weather_ml.py

4️⃣ Provide Input
Enter Humidity (%):
Enter Wind Speed (m/s):
The program will output the predicted temperature.





