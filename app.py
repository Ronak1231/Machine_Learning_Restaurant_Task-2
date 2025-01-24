# app.py
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Load the dataset
file_path = r"C:/Users/91982/Documents/Cognifyz Technologies/Task 2/Dataset.csv"  # Path to your dataset
dataset = pd.read_csv(file_path)

# Step 1: Preprocessing the Dataset
def preprocess_data():
    relevant_columns = ['Restaurant Name', 'City', 'Cuisines', 'Price range', 'Aggregate rating', 'Votes']
    filtered_data = dataset[relevant_columns].copy()
    filtered_data['Cuisines'] = filtered_data['Cuisines'].fillna('Unknown')
    filtered_data['Cuisines'] = filtered_data['Cuisines'].apply(lambda x: x.split(', '))

    mlb_cuisines = MultiLabelBinarizer()
    cuisines_encoded = pd.DataFrame(mlb_cuisines.fit_transform(filtered_data['Cuisines']),
                                     columns=mlb_cuisines.classes_,
                                     index=filtered_data.index)

    city_encoded = pd.get_dummies(filtered_data['City'], prefix='City')

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(filtered_data[['Price range', 'Aggregate rating', 'Votes']])
    scaled_features = pd.DataFrame(scaled_features, columns=['Price range', 'Aggregate rating', 'Votes'], index=filtered_data.index)

    preprocessed_data = pd.concat([filtered_data[['Restaurant Name', 'City']], city_encoded, cuisines_encoded, scaled_features], axis=1)
    return preprocessed_data, mlb_cuisines, scaler

preprocessed_data, mlb_cuisines, scaler = preprocess_data()

# Step 2: Content-Based Filtering
def recommend_restaurants(user_preferences, data, mlb_cuisines, scaler, top_n=5):
    user_preferences['Cuisines'] = [c.lower() for c in user_preferences['Cuisines']]
    user_vector = pd.DataFrame([user_preferences])
    user_vector[['Price range', 'Aggregate rating', 'Votes']] = scaler.transform(
        user_vector[['Price range', 'Aggregate rating', 'Votes']])

    try:
        user_cuisines = pd.DataFrame(mlb_cuisines.transform([user_vector['Cuisines'][0]]),
                                     columns=mlb_cuisines.classes_,
                                     index=user_vector.index)
    except ValueError:
        user_cuisines = pd.DataFrame(0, columns=mlb_cuisines.classes_, index=user_vector.index)

    user_city = pd.get_dummies(user_vector['City'], prefix='City')
    user_vector = pd.concat([user_city, user_cuisines, user_vector[['Price range', 'Aggregate rating', 'Votes']]], axis=1).fillna(0)
    user_vector = user_vector.reindex(columns=data.columns.drop(['Restaurant Name', 'City']), fill_value=0)

    similarity = cosine_similarity(user_vector, data.drop(columns=['Restaurant Name', 'City']))
    data['Similarity'] = similarity[0]
    return data.nlargest(top_n, 'Similarity')[['Restaurant Name', 'Similarity']]

# Step 3: Evaluate the System
def evaluate_system(data, mlb_cuisines, scaler):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    simulated_user = test.sample(1)
    cuisine_columns = mlb_cuisines.classes_
    user_cuisines = [cuisine for cuisine in cuisine_columns if simulated_user[cuisine].values[0] == 1]

    simulated_preferences = {
        'City': simulated_user['City'].values[0],
        'Cuisines': user_cuisines,
        'Price range': simulated_user['Price range'].values[0],
        'Aggregate rating': simulated_user['Aggregate rating'].values[0],
        'Votes': simulated_user['Votes'].values[0]
    }

    recommendations = recommend_restaurants(simulated_preferences, train, mlb_cuisines, scaler)
    
    # Normalize restaurant names for comparison
    recommendations['Restaurant Name'] = recommendations['Restaurant Name'].str.strip().str.lower()
    test['Restaurant Name'] = test['Restaurant Name'].str.strip().str.lower()

    matching_restaurants = test[test['Restaurant Name'].isin(recommendations['Restaurant Name'])]

    predicted_scores = recommendations[recommendations['Restaurant Name'].isin(matching_restaurants['Restaurant Name'])]['Similarity'].values
    actual_scores = matching_restaurants['Aggregate rating'].values

    # Check if predicted_scores and actual_scores are empty
    if len(predicted_scores) == 0 or len(actual_scores) == 0:
        return None  # Return None instead of inf

    min_length = min(len(predicted_scores), len(actual_scores))
    mse = mean_squared_error(actual_scores[:min_length], predicted_scores[:min_length])
    return mse

# Step 4: Visualization Generation
def generate_visualizations(filtered_data, user_preferences):
    # Check the columns of the DataFrame
    print("Columns in filtered_data:", filtered_data.columns.tolist())

    # Visualization 1: Distribution of Price Range
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Price range', data=filtered_data, palette='coolwarm')
    plt.title('Distribution of Price Range')
    plt.xlabel('Price Range')
    plt.ylabel('Count')
    plt.savefig('static/images/visualizations/price_range_distribution.png')
    plt.close()

    # Visualization 2: Distribution of Aggregate Ratings
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_data['Aggregate rating'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Aggregate Ratings')
    plt.xlabel('Aggregate Rating')
    plt.ylabel('Frequency')
    plt.savefig('static/images/visualizations/aggregate_rating_distribution.png')
    plt.close()

    # Visualization 3: Votes vs. Aggregate Rating
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Votes', y='Aggregate rating', data=filtered_data, color='orange')
    plt.title('Votes vs Aggregate Rating')
    plt.xlabel('Votes')
    plt.ylabel('Aggregate Rating')
    plt.savefig('static/images/visualizations/votes_vs_rating.png')
    plt.close()

    # Visualization 4: Top 10 Most Common Cuisines
    if 'Cuisines' in filtered_data.columns:
        top_cuisines = filtered_data['Cuisines'].explode().value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_cuisines.index, y=top_cuisines.values, palette='viridis')
        plt.title('Top 10 Most Common Cuisines')
        plt.xlabel('Cuisine')
        plt.ylabel('Count')
        plt.savefig('static/images/visualizations/top_cuisines.png')
        plt.xticks(rotation=45)
        plt.close()
    else:
        print("Warning: 'Cuisines' column not found in filtered_data.")

    # Step 5: Refine Correlation Analysis
    recommendations = recommend_restaurants(user_preferences, preprocessed_data, mlb_cuisines, scaler)

    # Merge the similarity scores with the numerical features of the recommended restaurants
    recommended_data = preprocessed_data[preprocessed_data['Restaurant Name'].isin(recommendations['Restaurant Name'])]
    recommended_data = recommended_data[['Restaurant Name', 'Price range', 'Aggregate rating', 'Votes']]
    
    # Ensure that the length of recommendations matches the recommended_data
    if len(recommendations) > 0:
        if len(recommended_data) == len(recommendations):
            recommended_data['Similarity'] = recommendations['Similarity'].values
        else:
            print(f"Warning: Length mismatch - recommended_data has {len(recommended_data)} rows but recommendations has {len(recommendations)} rows.")
            recommended_data['Similarity'] = np.nan  # Assign NaN if lengths do not match
    else:
        print("No recommendations found.")
        recommended_data['Similarity'] = np.nan  # Assign NaN if no recommendations

    # Compute the correlation matrix between numerical features and similarity scores
    correlation_matrix = recommended_data[['Price range', 'Aggregate rating', 'Votes', 'Similarity']].corr()

    # Plotting the correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features with Similarity Scores')
    plt.savefig('static/images/visualizations/correlation_heatmap.png')
    plt.close()

    # Visualization 6: Most Recommended Restaurants
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Similarity', y='Restaurant Name', data=recommendations.head(10), palette='coolwarm')
    plt.title('Top 10 Recommended Restaurants')
    plt.xlabel('Similarity Score')
    plt.ylabel('Restaurant Name')
    plt.savefig('static/images/visualizations/top_10_recommended_restaurants.png')
    plt.close()

# Step 5: Flask Routes
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('recommendation'))
        else:
            return "Invalid credentials. Please try again."
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)  # Use default hashing method
        new_user = User(email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Remove user_id from session
    return redirect(url_for('login'))  # Redirect to login page

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    recommendations = pd.DataFrame()  # Initialize recommendations as an empty DataFrame
    mse = None  # Initialize mse to None

    # Calculate MSE on a fixed test set
    mse = evaluate_system(preprocessed_data, mlb_cuisines, scaler)  # Calculate MSE independently

    if request.method == 'POST':
        user_city = request.form['city']
        user_cuisines = request.form['cuisines'].split(', ')
        user_price_range = int(request.form['price_range'])
        user_rating = float(request.form['rating'])
        user_votes = int(request.form['votes'])

        user_preferences = {
            'City': user_city,
            'Cuisines': [c.lower() for c in user_cuisines],
            'Price range': user_price_range,
            'Aggregate rating': user_rating,
            'Votes': user_votes
        }

        recommendations = recommend_restaurants(user_preferences, preprocessed_data, mlb_cuisines, scaler)
        generate_visualizations(preprocessed_data, user_preferences)  # Generate visualizations

    return render_template('index.html', recommendations=recommendations.to_dict(orient='records'), mse=mse)

@app.route('/visualizations')
def visualizations():
    visualizations = os.listdir('static/images/visualizations')  # List visualization images
    return render_template('visualizations.html', visualizations=visualizations)

if __name__ == '__main__':
    with app.app_context():  # Set up application context
        db.create_all()  # Create database tables
    app.run(debug=True)