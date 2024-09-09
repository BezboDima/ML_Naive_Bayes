import pandas as pd
import numpy as np
from Training import Training
from Testing import Testing
"""
# Set random seed for reproducibility
np.random.seed(10)

# Number of instances
n_instances = 100000

# Define possible values for each attribute
attribute1_bins = ['Low', 'Medium', 'High']
attribute2_bins = ['A', 'B', 'C']
attribute3_bins = ['X', 'Y', 'Z']

# Generate random categorical data
data = {
    'Attribute1': np.random.choice(attribute1_bins, n_instances),
    'Attribute2': np.random.choice(attribute2_bins, n_instances),
    'Attribute3': np.random.choice(attribute3_bins, n_instances),
    'Class': np.random.choice(['Class1', 'Class2', 'Class3'], n_instances)
}

tdata = {
    'Attribute1': np.random.choice(attribute1_bins, 10),
    'Attribute2': np.random.choice(attribute2_bins, 10),
    'Attribute3': np.random.choice(attribute3_bins, 10),
    'Class': np.random.choice(['Class1', 'Class2', 'Class3'], 10)
}
"""

df_covid = pd.read_csv("Cleaned-Data.csv").drop(columns=['Country'])

value_mapping = {
    'Severity_Mild': 0,
    'Severity_Moderate':1,
    'Severity_None':2,
    'Severity_Severe':3
}
# Create DataFrame

df_covid['Severity'] = df_covid.apply(lambda row: value_mapping[next(col for col in value_mapping if row[col] == 1)], axis=1)

df_covid.drop(columns=['Severity_Mild','Severity_Moderate', 'Severity_None','Severity_Severe'])

df = pd.DataFrame(df_covid.iloc[:int(len(df_covid) * 0.8)])

tdf = pd.DataFrame(df_covid.iloc[int(len(df_covid) * 0.8):])
"""
# Encode categorical attributes and target variable
df_encoded = df.copy()
for column in ['Outlook','Temperature','Humidity','Windy','Play']:
    df_encoded[column] = pd.Categorical(df_encoded[column]).codes

tdf_encoded = tdf.copy()
for column in ['Outlook','Temperature','Humidity','Windy','Play']:
    tdf_encoded[column] = pd.Categorical(tdf_encoded[column]).codes
"""

training_instance = Training(df, "Severity")

# print(training_instance.training_set)
training_instance.training_process()
classifier = training_instance.classifier

testing_instance = Testing(classifier=classifier, testing_set=tdf, target="Severity")

testing_instance.run_test()


print(testing_instance.get_precision())
print(testing_instance.get_recall())

