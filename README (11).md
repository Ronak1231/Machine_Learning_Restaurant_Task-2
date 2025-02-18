# 🍽️ Machine Learning Restaurant Task - End-to-End ML Application

This project implements an **end-to-end machine learning system** for restaurant data analysis using **Flask** for the user interface and **Scikit-Learn** for predictive modeling. It helps analyze restaurant data and make insightful predictions.

---

## ✅ Features

- **Data Preprocessing**: Cleans missing values and selects key features.
- **Machine Learning Models**: Implements **Decision Tree, Random Forest, and Logistic Regression**.
- **Web Application**: Built using **Flask** to interact with the model.
- **Performance Evaluation**: Uses **accuracy, precision, and recall metrics**.
- **Data Visualization**: Displays insights from the dataset.
- **User Interaction**: Users can input restaurant details and get predictions.

---

## 📜 Prerequisites

Ensure the following are installed:

1. **Python 3.8 or above**  
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scriptsctivate
   ```
2. **Required Libraries** (Install from `requirements.txt`):
   ```sh
   pip install -r requirements.txt
   ```
3. **Dataset**: Ensure `Dataset.csv` is in the project directory.

---

## 🛠 Setup Instructions

### 1. Clone the Repository

```sh
git clone https://github.com/Ronak1231/Machine_Learning_Restaurant_Task-2.git
cd Machine_Learning_Restaurant_Task-2
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Run the Flask Application

```sh
python app.py
```

---

## 🗃️ File Structure

```
Machine_Learning_Restaurant_Task-2/

├── Dataset.csv                     # Dataset file
├── Static
│   ├── Images
|        ├── Visualization
│   ├── css
|        ├── style.css
│   ├── js
|        ├── script.js
|
├── Trail
│   ├── Restaurant_Task-2
|
├── templates
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── visualization.html
|
├── app.py
├── requirements.txt
├── LICENSE
├── README.md                                 # Project documentation
```

---

## 🤷 How It Works

### Backend Flow

1. **Data Loading**: Reads and processes dataset, handling missing values.
2. **Feature Engineering**: Identifies key attributes relevant for prediction.
3. **Model Training**: Implements **Decision Tree, Random Forest, and Logistic Regression**.
4. **Model Evaluation**: Uses accuracy, precision, and recall for performance measurement.
5. **Prediction**: Generates insights from restaurant data.

### Frontend Flow

- Users **input restaurant details** via **Flask UI**.
- The system processes the input and **visualizes data trends**.
- The model predicts and returns the result.

---

## 🤖 Technologies Used

- **Python**: Programming language for data processing and ML.
- **Flask**: Web-based application interface.
- **Scikit-Learn**: Machine learning models and data preprocessing.
- **Matplotlib & Seaborn**: Data visualization tools.

---

## 🚚 Deployment

This project can be deployed on **AWS, Google Cloud, or Heroku**. Ensure API keys and environment variables are configured properly.

---

## 🔜 Future Improvements

1. Add real-time restaurant data integration.
2. Implement deep learning models for enhanced accuracy.
3. Expand the web app with additional features.

---

## 🤝 Acknowledgments

- [Scikit-Learn](https://scikit-learn.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Matplotlib](https://matplotlib.org/)

---

## ✍️ Author  
[Ronak Bansal](https://github.com/Ronak1231)

---

## 🙌 Contributing  
Feel free to fork this repository, improve it, and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🐛 Troubleshooting  
If you encounter issues, create an issue in this repository.

---

## 📧 Contact  
For inquiries or support, contact [ronakbansal12345@gmail.com].
