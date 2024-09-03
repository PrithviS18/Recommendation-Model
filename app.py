
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Sample alumni dataset
alumni_data = {
    'alumni_id': ['66d4316ac746b56f811d68a9', '66d43194c746b56f811d68ad', '66d43230c746b56f811d68b1', '66d4328ec746b56f811d68b5','66d432d0c746b56f811d68b9'],
    'interests': [
        'once a student, always a proud alumnus,coding, gaming, web dev, app dev,cst , aiml',
        'coding, web dev, cst, aiml',
        'gaming, app dev, cst, aiml',
        'cricket, c++, cst, aiml',
        'coding ,gaming, web dev, app dev, cst, aiml'
    ]
}

alumni_df = pd.DataFrame(alumni_data)

# Sample student dataset
student_data = {
    'student_id': ['66d42d0bc908db093154dc57','66d42d8bc908db093154dc5f','66d42dcfc908db093154dc63','66d42e47c908db093154dc67','66d42e90c908db093154dc6b'],
    'interests': [
        'aspiring artist , eternal student, coding, gaming, web dev, app dev, cst, aiml',
        'aspiring artist , eternal student, coding, web dev, cst, aiml',
        'aspiring artist , eternal student, gaming, app dev, cst, aiml',
        'aspiring artist , eternal student, cricket, c++, cst , aiml',
        'aspiring artist , eternal student,coding , gaming , web dev, app dev, cst , aiml'
    ]
}

student_df = pd.DataFrame(student_data)

# Convert interests to TF-IDF features for both alumni and students
vectorizer = TfidfVectorizer()

# Fit the vectorizer on alumni interests and transform both alumni and student interests
alumni_tfidf = vectorizer.fit_transform(alumni_df['interests'])
student_tfidf = vectorizer.transform(student_df['interests'])

# Function to compute recommendations for each student
def recommend_for_students(student_id, student_tfidf, alumni_tfidf, alumni_df):
    try:
        # Find the index of the student in the DataFrame
        student_index = student_df.index[student_df['student_id'] == student_id].tolist()[0]

        # Compute cosine similarity between the current student and all alumni
        similarity_scores = cosine_similarity(student_tfidf[student_index], alumni_tfidf).flatten()

        # Create a DataFrame for similarity scores
        alumni_df['similarity'] = similarity_scores

        # Get the top N alumni based on similarity
        top_alumni = alumni_df.sort_values(by='similarity', ascending=False).head(3)

        # Store the recommendations in a list
        recommendations = top_alumni[['alumni_id', 'interests', 'similarity']].to_dict(orient='records')
        return recommendations
    
    except IndexError:
        return []
@app.route('/')
def home():
    return "Flask API is running!"
    
@app.route('/recommend/<student_id>', methods=['GET'])
def recommend(student_id):
    try:
        recommendations = recommend_for_students(student_id, student_tfidf, alumni_tfidf, alumni_df)
        if recommendations:
            return jsonify(recommendations)
        else:
            return jsonify({"error": "Student ID not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
