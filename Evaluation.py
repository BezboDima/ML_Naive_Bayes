import numpy as np
from DataControl import DataControl
from Training import Training
from Testing import Testing

class Evaluation:
    def __init__(self, dataset:DataControl, target: str, folds: int = 10):
        self.dataset = dataset
        self.folds = folds
        self.target = target
        self.recalls = []
        self.precisions = []
    def run_eval(self):

        split = 1/self.folds
        start = 0.0
        finish = split

        while finish < 1:
            self.dataset.train_test_split(split_start=start, split_end=finish)

            training_instance = Training(self.dataset.training, target=self.target)
            training_instance.training_process()
            classifier = training_instance.get_classifier()

            testing_instance = Testing(classifier=classifier, testing_set=self.dataset.testing, target=self.target)
            testing_instance.run_test()

            print("Training\n", self.dataset.training)
            print("Testing\n", self.dataset.testing)
            print("Recall: " , testing_instance.get_recall())
            print("Precision: " , testing_instance.get_precision())

            self.recalls.append(testing_instance.get_recall())
            self.precisions.append(testing_instance.get_precision())

            start += split
            finish += split

    def get_avg_recall(self):
        return np.mean(self.recalls)
    
    def get_avg_precision(self):
        return np.mean(self.precisions)




