import os
from flask import Flask
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from collections import defaultdict


def generate_random_feature(range, num_features, p):
    return np.random.choice(range, num_features, p)


def simulate_missing_data(university, degree):
    example = defaultdict()
    example['university'] = university
    example['degree'] = degree

    median_salary = pd.read_csv('median_entry_salary_by_degree.csv')
    median_entry_uni_degree_salary = median_salary[
        (median_salary['university'] == university) & (median_salary['degree'] == degree)].median_entry_salary.values[0]
    example['median_entry_salary'] = median_entry_uni_degree_salary

    # numpy.random.seed(int(time.time()))

    # Simulate the chance of being a gold/silver/brown medal winner of an international olympiad (e.g. in Math)
    # TODO: Simulate that for all possible subjects
    # TODO: Use a count feature, rather than a boolean (e.g. count the number of gold medals in x olympiad of y type)
    example['has_gold_medals_in_international_olympiads'] = generate_random_feature(2, 1, p=[0.99999, 0.00001])[0]
    example['has_silver_medals_in_international_olympiads'] = generate_random_feature(2, 1, p=[0.99998, 0.00002])[0]
    example['has_brown_medals_in_international_olympiads'] = generate_random_feature(2, 1, p=[0.99997, 0.00003])[0]
    example['was_in_finals_in_international_olympiads'] = generate_random_feature(2, 1, p=[0.99995, 0.00005])[0]

    example['has_gold_medals_in_national_olympiads'] = generate_random_feature(2, 1, p=[0.9999, 0.0001])[0]
    example['has_silver_medals_in_national_olympiads'] = generate_random_feature(2, 1, p=[0.9998, 0.0002])[0]
    example['has_brown_medals_in_national_olympiads'] = generate_random_feature(2, 1, p=[0.9997, 0.0003])[0]
    example['was_in_finals_in_national_olympiads'] = generate_random_feature(2, 1, p=[0.9995, 0.0005])[0]

    example['has_gold_medals_in_regional_olympiads'] = generate_random_feature(2, 1, p=[0.999, 0.001])[0]
    example['has_silver_medals_in_regional_olympiads'] = generate_random_feature(2, 1, p=[0.998, 0.002])[0]
    example['has_brown_medals_in_regional_olympiads'] = generate_random_feature(2, 1, p=[0.997, 0.003])[0]
    example['was_in_finals_in_regional_olympiads'] = generate_random_feature(2, 1, p=[0.995, 0.005])[0]

    example['has_gold_medal_winner_in_international_sports_competition'] = generate_random_feature(2, 1, p=[0.99999, 0.00001])[0]
    example['has_silver_medals_in_international_sports_competition'] = generate_random_feature(2, 1, p=[0.99998, 0.00002])[0]
    example['has_brown_medals_in_international_sports_competition'] = generate_random_feature(2, 1, p=[0.99997, 0.00003])[0]
    example['was_in_finals_in_international_sports_competition'] = generate_random_feature(2, 1, p=[0.99995, 0.00005])[0]

    example['has_gold_medals_in_national_sports_competition'] = generate_random_feature(2, 1, p=[0.9999, 0.0001])[0]
    example['has_silver_medals_in_national_sports_competition'] = generate_random_feature(2, 1, p=[0.9998, 0.0002])[0]
    example['has_brown_medals_in_national_sports_competition'] = generate_random_feature(2, 1, p=[0.9997, 0.0003])[0]
    example['was_in_finals_in_national_sports_competition'] = generate_random_feature(2, 1, p=[0.9995, 0.0005])[0]

    example['has_gold_medals_in_regional_sports_competition'] = generate_random_feature(2, 1, p=[0.999, 0.001])[0]
    example['has_silver_medals_in_regional_sports_competition'] = generate_random_feature(2, 1, p=[0.998, 0.002])[0]
    example['has_brown_medals_in_regional_sports_competition'] = generate_random_feature(2, 1, p=[0.997, 0.003])[0]
    example['was_in_finals_in_regional_sports_competition'] = generate_random_feature(2, 1, p=[0.995, 0.005])[0]

    example['num_years_high_school_president'] = generate_random_feature(4, 1, p=[0.88, 0.06, 0.03, 0.03])[0]
    example['num_years_high_school_class_president'] = generate_random_feature(4, 1, p=[0.60, 0.20, 0.10, 0.10])[0]

    example['has_fluent_in_english'] = generate_random_feature(2, 1, p=[0.80, 0.20])[0]
    example['has_fluent_in_spanish'] = generate_random_feature(2, 1, p=[0.85, 0.15])[0]
    example['has_fluent_in_german'] = generate_random_feature(2, 1, p=[0.92, 0.08])[0]
    example['has_fluent_in_chinese'] = generate_random_feature(2, 1, p=[0.85, 0.15])[0]
    example['has_fluent_in_french'] = generate_random_feature(2, 1, p=[0.95, 0.05])[0]

    example['num_papers_published'] = generate_random_feature(4, 1, p=[0.96, 0.02, 0.01, 0.01])[0]

    example['num_months_of_work_experience'] = generate_random_feature(10, 1, p=[0.05, 0.20, 0.20, 0.15,
                                                                                 0.10, 0.10, 0.07, 0.07, 0.03, 0.03])
    example_df = pd.DataFrame(example, index=[0])
    return example_df


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_pyfile('config.py', silent=True)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    model = joblib.load('trained_random_forest.joblib')

    # a simple page that says hello
    @app.route('/predict_salary')
    def predict_salary(request, methods=['POST']):
        university = request.form['university']
        degree = request.form['degree']

        features = simulate_missing_data(university, degree)

        features = pd.get_dummies(features)
        features = np.array(features)

        features['university_' + university] = 1
        features['degree_' + degree] = 1

        predicted_entry_salary = model.predict(features)


        return predicted_entry_salary

    return app