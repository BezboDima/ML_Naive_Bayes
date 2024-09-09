import pandas as pd
from typing import List
import numpy as np
import random
import math


class DataControl:
    def __init__(self, file_path: str, columns: List[str]):
        read = pd.read_csv(file_path, header=None)
        self.data_table = read.sample(frac=1.0, random_state=43).reset_index(drop=True)
        self.data_table.columns = columns
        self.data_processed = self.data_table.copy()

        self.training = None
        self.testing = None

    def hot_encoding_categorical(self, columns: List[str]) -> None:
        for column in columns:
            self.data_processed[column] = pd.Categorical(self.data_table[column]).codes

    def train_test_split(self, split_start: float, split_end: float):
        self.training = pd.DataFrame(self.data_processed.iloc[:int(len(self.data_processed) * split_start)])
        self.testing = pd.DataFrame(self.data_processed.iloc[int(len(self.data_processed) * split_start):int(len(self.data_processed) * split_end)])
        self.training = pd.concat([self.training, pd.DataFrame(self.data_processed.iloc[int(len(self.data_processed) * split_end):])], ignore_index=True)


    def noise(self, target: str) -> None:
        features = list(self.data_processed.drop(columns=[target]))
        affected = []
        for _ in range(math.ceil(len(features)/10)):
            affected = features.pop(random.randrange(len(features))) 
        for column in affected:
            np.random.permutation(self.data_processed[column])

    def continious_to_categorical(self, columns: List[str]):
        for column in columns:
            self.data_processed[column] = pd.qcut(self.data_table[column], q=[0, .2, .4, .6, .8, 1], labels=range(1,6))

    def drop_repeating_data(self):
        for column in self.data_table.columns.to_list():
            if (len(self.data_table[column].unique()) == 1):
                self.data_processed = self.data_table.drop(columns=[column])

    def handle_missing_values(self, column):
        self.data_processed[column].fillna(self.data_processed[column].median(), inplace=True)




