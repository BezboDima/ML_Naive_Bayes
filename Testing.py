import pandas as pd
import numpy as np
from Classifier import Classifier

class Testing:
    def __init__(self, classifier:Classifier, testing_set:pd.DataFrame, target: str):
        self.classifier = classifier
        self.testing_set = testing_set.drop(columns=[target])
        self.target = testing_set[target]
        self.recall = None
        self.precision = None
    def run_test(self):
        
        self.classifier.make_prediction(self.testing_set)
        predictions = self.classifier.predictions
        n = len(self.classifier.class_prob)
        label_matrix = np.zeros((n, n))

        print("targ", self.target.to_list())
        print("pred" , predictions)
        for pred, true in zip(predictions, self.target):
            label_matrix[pred][true] += 1

        print(label_matrix)
        precision = [0] * n
        for i in range(len(label_matrix)):
            total = 0
            for j in range(len(label_matrix[0])):
                total += label_matrix[i][j]
                if i == j:
                    correct = label_matrix[i][j]
            precision[i] = correct / total
        self.precision = sum(precision) / n


        recalls = [0] * n
        for i in range(len(label_matrix[0])):
            total = 0
            for j in range(len(label_matrix)):
                total += label_matrix[j][i]
                if i == j:
                    correct = label_matrix[j][i]
            recalls[i] = correct / total

        self.recall = sum(recalls) / n

    def get_precision(self):
        return self.precision
    def get_recall(self):
        return self.recall