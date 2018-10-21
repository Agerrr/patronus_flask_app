import os
from flask import Flask
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
import json

sorted_feature_names = ["degree_Associate's Degree", 'degree_Bachelor of Accountancy (BAcc)', 'degree_Bachelor of Architecture (BArch)', 'degree_Bachelor of Arts (BA)', 'degree_Bachelor of Arts (BA), Business Management', 'degree_Bachelor of Arts (BA), English Literature', 'degree_Bachelor of Arts (BA), Event Management', 'degree_Bachelor of Arts (BA), Graphic Design', 'degree_Bachelor of Arts (BA), History', 'degree_Bachelor of Arts (BA), Public Relations (PR)', 'degree_Bachelor of Arts (BA), Theatre & Drama Studies', 'degree_Bachelor of Arts and Science (BASc)', 'degree_Bachelor of Business (BB)', 'degree_Bachelor of Business Administration (BBA)', 'degree_Bachelor of Business Studies (BBS)', 'degree_Bachelor of Civil Law (BCL)', 'degree_Bachelor of Commerce (BCom)', 'degree_Bachelor of Design (BDes)', 'degree_Bachelor of Education (BEd)', 'degree_Bachelor of Engineering (BEng / BE)', 'degree_Bachelor of Engineering (BEng / BE), Aeronautical Engineering', 'degree_Bachelor of Engineering (BEng / BE), Civil Engineering (CE)', 'degree_Bachelor of Engineering (BEng / BE), Mechanical Engineering (ME)', 'degree_Bachelor of Engineering (BEng / BE), Software Engineering', 'degree_Bachelor of Finance (BFin)', 'degree_Bachelor of Fine Arts (BFA)', 'degree_Bachelor of Humanities (BHum)', 'degree_Bachelor of Industrial Design (BID)', 'degree_Bachelor of Landscape Architecture (BLArch)', 'degree_Bachelor of Laws (LLB)', 'degree_Bachelor of Liberal Arts (BLA)', 'degree_Bachelor of Music (BMus / BM)', 'degree_Bachelor of Philosophy (BPh / PhB)', 'degree_Bachelor of Science (BS / BSc)', 'degree_Bachelor of Science (BS / BSc), Applied Mathematics', 'degree_Bachelor of Science (BS / BSc), Biomedical Sciences', 'degree_Bachelor of Science (BS / BSc), Building Surveying', 'degree_Bachelor of Science (BS / BSc), Computer Science (CS)', 'degree_Bachelor of Science (BS / BSc), Construction Project Management', 'degree_Bachelor of Science (BS / BSc), Economics', 'degree_Bachelor of Science (BS / BSc), Geography', 'degree_Bachelor of Science (BS / BSc), Mathematics', 'degree_Bachelor of Science (BS / BSc), Multimedia Computing', 'degree_Bachelor of Science (BS / BSc), Product Design', 'degree_Bachelor of Science (BS / BSc), Quantity Surveying', 'degree_Bachelor of Science, Electrical Engineering (BSEE)', 'degree_Bachelor of Science, Mechanical Engineering (BSME)', 'degree_Bachelor of Social Work (BSW / BSocW)', 'degree_Bachelor of Technology (BT / BTech)', 'degree_Bachelor of Theology (BTh)', "degree_Bachelor's Degree", "degree_Bachelor's Degree, Business Management", "degree_Bachelor's Degree, Interior Design", "degree_Bachelor's Degree, Software Engineering", 'degree_Certificate (Cert)', 'degree_Certificate (Cert), Civil Engineering (CE)', 'degree_Certificate (Cert), Psychology & Sociology', 'degree_Certificate (Cert), Retail Management', 'degree_Doctor of Philosophy (PhD)', 'degree_Doctor of Science (DS)', 'degree_Doctorate (PhD)', 'degree_Doctorate (PhD), Mechanical Engineering (ME)', 'degree_Executive Masters of Business Administration (Exec MBA)', 'degree_Executive Masters of Business Administration (Exec MBA), Business Management', 'degree_Executive Masters of Business Administration (Exec MBA), Finance', 'degree_Executive Masters of Business Administration (Exec MBA), Strategy', 'degree_Graduate Certificate', 'degree_Graduate Certificate, Psychology', 'degree_Higher National Diploma (HND)', 'degree_Juris Doctor (JD), Master of Jurisprudence, or Master of Law (LLM)', 'degree_Master of Architecture (MArch)', 'degree_Master of Arts (MA)', 'degree_Master of Arts (MA), Computer Animation', 'degree_Master of Arts (MA), English Literature', 'degree_Master of Business Administration (MBA)', 'degree_Master of Business Administration (MBA), Accounting & Finance', 'degree_Master of Business Administration (MBA), Business & Marketing', 'degree_Master of Business Administration (MBA), Business Administration', 'degree_Master of Business Administration (MBA), Business Management', 'degree_Master of Business Administration (MBA), Finance', 'degree_Master of Business Administration (MBA), General Business', 'degree_Master of Business Administration (MBA), Innovation Management', 'degree_Master of Business Administration (MBA), International Marketing', 'degree_Master of Business Administration (MBA), Marketing', 'degree_Master of Business Administration (MBA), Project Management', 'degree_Master of Business Administration (MBA), Strategy', 'degree_Master of Civil Engineering (MCE)', 'degree_Master of Computer Science (MCS)', 'degree_Master of Design (MDes)', 'degree_Master of Education (MEd)', 'degree_Master of Engineering (MEng / ME)', 'degree_Master of Engineering (MEng / ME), Mechanical Engineering (ME)', 'degree_Master of Engineering Management (MEM)', 'degree_Master of Environmental Science (MESc)', 'degree_Master of Human Services (MHS)', 'degree_Master of Information Science (MIS)', 'degree_Master of Journalism (MJ)', 'degree_Master of Landscape Architecture (MLA)', 'degree_Master of Liberal Arts (MLA)', 'degree_Master of Management (MMgt / MM)', 'degree_Master of Mechanical Engineering (MME)', 'degree_Master of Medical Science (MMS / MMSc)', 'degree_Master of Philosophy (MPhil)', 'degree_Master of Public Health (MPH)', 'degree_Master of Public Policy (MPP)', 'degree_Master of Regional Planning (MRP)', 'degree_Master of Science (MS)', 'degree_Master of Science (MS), Mathematics', "degree_Master's Degree", "degree_Master's Degree (non-MBA)", 'degree_Post Graduate Certificate', 'degree_Post Graduate Diploma', 'has_brown_medals_in_international_olympiads', 'has_brown_medals_in_international_sports_competition', 'has_brown_medals_in_national_olympiads', 'has_brown_medals_in_national_sports_competition', 'has_brown_medals_in_regional_olympiads', 'has_brown_medals_in_regional_sports_competition', 'has_fluent_in_chinese', 'has_fluent_in_english', 'has_fluent_in_french', 'has_fluent_in_german', 'has_fluent_in_spanish', 'has_gold_medal_winner_in_international_sports_competition', 'has_gold_medals_in_international_olympiads', 'has_gold_medals_in_national_olympiads', 'has_gold_medals_in_national_sports_competition', 'has_gold_medals_in_regional_olympiads', 'has_gold_medals_in_regional_sports_competition', 'has_silver_medals_in_international_olympiads', 'has_silver_medals_in_international_sports_competition', 'has_silver_medals_in_national_olympiads', 'has_silver_medals_in_national_sports_competition', 'has_silver_medals_in_regional_olympiads', 'has_silver_medals_in_regional_sports_competition', 'median_entry_salary', 'num_months_of_work_experience', 'num_papers_published', 'num_years_high_school_class_president', 'num_years_high_school_president', 'university_Anglia Ruskin University', 'university_Aston University', 'university_Bournemouth University', 'university_Bristol University', 'university_Brunel University, London', 'university_Cardiff University', 'university_City University - London', 'university_Coventry University', 'university_Cranfield University', 'university_De Montfort University - Leicester', 'university_Glasgow Caledonian University', 'university_Imperial College, London', "university_King's College London", 'university_Kingston University', 'university_Leeds Metropolitan University', 'university_Liverpool John Moores University', 'university_London Metropolitan University', 'university_London South Bank University', 'university_Loughborough University', 'university_Middlesex University', 'university_Napier University (Edinburgh)', 'university_Open University - England', 'university_Oxford Brookes University', 'university_Oxford University', 'university_Sheffield Hallam University', 'university_Southampton Solent University', 'university_Staffordshire University', 'university_The Manchester Metropolitan University', 'university_The Nottingham Trent University', 'university_The Robert Gordon University', 'university_The University Of Central Lancashire', 'university_The University of Brighton', 'university_The University of Cambridge', 'university_The University of East London', 'university_The University of Exeter', 'university_The University of Greenwich', 'university_The University of Huddersfield', 'university_The University of Hull', 'university_The University of Lancaster', 'university_The University of Leeds', 'university_The University of Leicester', 'university_The University of Lincoln', 'university_The University of Manchester', 'university_The University of Newcastle-upon-Tyne', 'university_The University of Northumbria at Newcastle', 'university_The University of Nottingham', 'university_The University of Portsmouth', 'university_The University of Reading', 'university_The University of Salford', 'university_The University of Strathclyde', 'university_The University of Teesside', 'university_The University of Wolverhampton', 'university_University College London', 'university_University Of Hertfordshire', 'university_University of Bath', 'university_University of Birmingham - United Kingdom', 'university_University of Derby', 'university_University of Durham', 'university_University of East Anglia', 'university_University of Edinburgh', 'university_University of Glamorgan', 'university_University of Glasgow', 'university_University of Kent', 'university_University of Liverpool', 'university_University of London', 'university_University of Plymouth', 'university_University of Sheffield', 'university_University of Southampton', 'university_University of Ulster', 'university_University of Warwick', 'university_University of Westminster', 'university_University of the West of England, Bristol', 'was_in_finals_in_international_olympiads', 'was_in_finals_in_international_sports_competition', 'was_in_finals_in_national_olympiads', 'was_in_finals_in_national_sports_competition', 'was_in_finals_in_regional_olympiads', 'was_in_finals_in_regional_sports_competition']


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

    model = joblib.load('/Users/agnieszkaszefer/dev/private/patronus_flask_app/trained_random_forest.joblib')
    median_salary_data = pd.read_csv('/Users/agnieszkaszefer/dev/private/patronus_flask_app/median_entry_salary_by_degree.csv')
    unique_unis = median_salary_data.university.unique().tolist()
    unique_degrees = median_salary_data.degree.unique().tolist()

    # a simple page that says hello
    @app.route('/predict_salary')
    def predict_salary(request, methods=['POST']):
        university = request.form['university']
        degree = request.form['degree']

        features = simulate_missing_data(university, degree)
        features = pd.get_dummies(features)

        for u in unique_unis:
            features['university_' + str(u)] = 0

        for d in unique_degrees:
            features['degree_' + str(d)] = 0

        features['university_' + str(university)] = 1
        features['degree_' + str(degree)] = 1

        features = features[sorted_feature_names]
        features = np.array(features)

        predicted_entry_salary = model.predict(features)

        predicted_salary_json = json.dumps({'predicted_salary': predicted_entry_salary[0]})
        return predicted_salary_json

    return app