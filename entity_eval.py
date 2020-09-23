import numpy as np
from sklearn.metrics import f1_score

class EntityEvaluator:
    def __init__(self):
        self.preds = []
        self.labels = []
    
    def merge_input(self, score_matrix, score_mask):
        """
        Args:
            score_matrix: [batch_size, num_classes]
            labels: [batch_size, num_classes]
        """
        if score_matrix.shape != score_mask.shape:
            print('size not match')
            return

        for i in range(score_matrix.shape[0]):
            score_idx = np.argmax(score_matrix[i])
            self.preds.append(score_idx + 1)
            mask_idx = np.argmax(score_mask[i])
            self.labels.append(mask_idx + 1)
    
    def calc_f1(self):
        # f1_score = 2 * (precision * recall) / (precision + recall)
        print('evaluate length: {}'.format(len(self.labels)))
        f1 = f1_score(y_true=self.labels, y_pred=self.preds, average='weighted')
        return f1
    
    def calc_accuracy(self):
        acc = [1 if label == pred else 0 for label, pred in zip(self.labels, self.preds)]
        return float(sum(acc)) / float(len(self.labels))

if __name__ == '__main__':
    score_matrix = np.array([
        [0,0,0,2],
        [2,1,5,2],
        [3,4,1,5]
    ])
    score_mask = np.array([
        [0,0,1,0],
        [2,1,5,2],
        [3,4,1,5]
    ])

    entity_evaluator = EntityEvaluator()
    entity_evaluator.merge_input(score_matrix, score_mask)
    f1_score = entity_evaluator.calc_f1()
    acc = entity_evaluator.calc_accuracy()
    print('f1 score: {}'.format(f1_score))
    print('accuracy: {}'.format(acc))