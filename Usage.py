import pandas as pd
import jellyfish
from fuzzywuzzy import fuzz
import joblib


def predict_name_matches(dataset_file, query, model_file):
    # Read the dataset
    dfDataset = pd.read_csv(dataset_file)

    # Add query column to dataframe
    dfDataset['Query'] = query

    # Compute Levenshtein distance
    levenshtein_distances = []
    for name, query in zip(dfDataset['Names'], dfDataset['Query']):
        levenshtein_distance = jellyfish.levenshtein_distance(name, query)
        levenshtein_distances.append(levenshtein_distance)

    # Add Levenshtein distance to the DataFrame
    dfDataset['LevenshteinDistance'] = levenshtein_distances

    # Compute fuzzywuzzy percentages
    fuzzy_percentages = []
    for name, query in zip(dfDataset['Names'], dfDataset['Query']):
        fuzzy_percentage = fuzz.ratio(name, query)
        fuzzy_percentages.append(fuzzy_percentage)

    # Add fuzzywuzzy percentages to the DataFrame
    dfDataset['FuzzyPercentages'] = fuzzy_percentages

    # Load the model
    model = joblib.load(model_file)

    # Prepare features for prediction
    XFit = dfDataset[['LevenshteinDistance', 'FuzzyPercentages']]

    # Predict name matches
    predictions = model.predict_proba(XFit)[:, 1]

    # Add predictions to the dataframe
    dfDataset['Predictions'] = predictions

    # Sort the dataframe by predictions in descending order
    sorted_df = dfDataset.sort_values(by=['Predictions'], ascending=False)

    return sorted_df


# Usage example
dataset_file = 'IndependentDS.csv'
query = 'Fred'
model_file = 'name_match_logistic_regression_model.pkl'

result_df = predict_name_matches(dataset_file, query, model_file)
print(result_df)
