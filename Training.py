import pandas as pd
from typing import Dict, List 
from Classifier import Classifier

df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [1, 0, 1]
})

class Training():

    def __init__(self, training_set:pd.DataFrame, target:str):
        self.training_set = training_set
        self.classifier = None
        self.target = target


    def training_process(self):
        class_probabilities = self._get_class_probabilities()
        print(class_probabilities)
        class_frames = self._separate_df_by_class()
        print(class_frames)
        distributions_A = self._get_probabilities_distributions(class_frames)
        print(distributions_A)
        
        self.classifier = Classifier(distributions_A, class_probabilities)

    def get_classifier(self):
        
        return self.classifier

    def _get_class_probabilities(self) -> Dict[int, int]:
        class_prob = {}
        class_count = self.training_set[self.target].value_counts()
        n = len(self.training_set)
        for key, val in class_count.items():
            class_prob[key] = val / n
        return class_prob
    
    def _separate_df_by_class(self) -> Dict[int, pd.DataFrame]:
        classes = self.training_set[self.target].unique()
        class_df_dic = {}
        for class_ in classes:
            separated = self.training_set[self.training_set[self.target] == class_]
            class_df_dic[class_] = separated.drop(columns=[self.target])
        return class_df_dic
    
    def _get_probabilities_distributions(self, class_spesific_sets: Dict[int, pd.DataFrame]):
        full_prob_distrtibution = {}
        for class_, dataset in class_spesific_sets.items():
            attributes_distribution = {}
            n_attributes = len(dataset.columns)
            for attribute in dataset.columns.tolist():
                attributes_distribution[attribute] = self._get_prob_distribution_attribute(dataset, attribute, n_attributes)
            full_prob_distrtibution[class_] = attributes_distribution    
        return full_prob_distrtibution
    
    def _get_prob_distribution_attribute(self, given_class: pd.DataFrame, attribute: str, n_attributes: int):
        attribute_count = given_class[attribute].value_counts()
        n_rows = len(given_class)
        prob_distribution_attribute = {}
        for key, val in attribute_count.items():
            prob_distribution_attribute[key] = (val + 1) / (n_rows + n_attributes)
        return prob_distribution_attribute
    
    """def _get_main_function(distributions_A: Dict[int, Dict[int, Dict[int, int]]], class_count: Dict[int, int]):
        main_dictionary = {}
        for class_, attributes in distributions_A.items():
            for attribute in class_:
                
                for instance in attribute
                """

            
            
            

