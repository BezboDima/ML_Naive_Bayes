from typing import Dict 
import pandas as pd

class Classifier():
    def __init__(self, distributions: Dict[int, Dict[int, float]], class_prob: Dict[int, float]) -> None:
        self.distributions = distributions
        self.class_prob = class_prob
        self.predictions = None
    def make_prediction(self, data: pd.DataFrame) -> None:

        self.predictions = []
        
        for row in range(data.index[0], data.index[-1] + 1):
            prediction = self._classify_row(data.loc[row].to_dict())
            self.predictions.append(prediction)

    def _classify_row(self, row: Dict[str, int]) -> int:
        class_predictions = {}
        
        for cl, distribution in self.distributions.items():

            product = 1
            for attribute, val in row.items():
                f_a_c = distribution[attribute].get(val, 0)
                product *= f_a_c
            
            class_predictions[cl] = product * self.class_prob[cl]
        return max(class_predictions, key=class_predictions.get)
