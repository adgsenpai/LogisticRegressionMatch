import numpy as np
from sklearn.linear_model import LogisticRegression
import jellyfish

# Function to process the name
def process_name(name):
    return name.upper()

# Function to check if two names match using logistic regression
def match_names(name1, name2):
    # Preprocess the names
    processed_name1 = process_name(name1)
    processed_name2 = process_name(name2)

    # Calculate additional features
    num_matching_chars = sum(c1 == c2 for c1, c2 in zip(processed_name1, processed_name2))
    levenshtein_distance = jellyfish.levenshtein_distance(processed_name1, processed_name2)

    # Create feature vectors for logistic regression
    feature_vector = np.array([
        len(processed_name1),          # Length of name 1
        len(processed_name2),          # Length of name 2
        int(processed_name1 == processed_name2),   # Whether names are the same
        num_matching_chars,            # Number of matching characters
        levenshtein_distance           # Levenshtein distance
    ]).reshape(1, -1)

    # Train a logistic regression model (or load a pre-trained model)
    # Replace with your training data
    X = np.array([[5, 5, 1, 5, 0], [4, 4, 0, 3, 1]])
    y = np.array([1, 0])  # Replace with your target labels
    model = LogisticRegression()
    model.fit(X, y)

    # Predict the match probability using the trained model
    match_probability = model.predict_proba(feature_vector)[:, 1]

    return round(match_probability[0], 2)

# Example usage
name1 = "John Smith"
name2 = "John Smith"
probability = match_names(name1, name2)
print("Name Match Probability for {} and {}: {}%".format(name1, name2, round(probability * 100, 2)))

name3 = "Ashlin Darius Govindasamy"
name4 = "Ashlin Govindasamy"
probability = match_names(name3, name4)
print("Name Match Probability for {} and {}: {}%".format(name3, name4, round(probability * 100, 2)))

name5 = "Emily Johnson"
name6 = "Emma Johnson"
probability = match_names(name5, name6)
print("Name Match Probability for {} and {}: {}%".format(name5, name6, round(probability * 100, 2)))

name7 = "Michael Brown"
name8 = "Mike Brown"
probability = match_names(name7, name8)
print("Name Match Probability for {} and {}: {}%".format(name7, name8, round(probability * 100, 2)))
