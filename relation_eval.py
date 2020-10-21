import numpy as np
from sklearn.metrics import f1_score

class RelationEvaluator:
    def __init__(self):
        self.preds = []
        self.labels = []
    
    def merge_input(self, score_tensor, label_mask):
        """
        Args:
            score_tensor: [k, k, num_classes]
            label_mask: [k, k, num_classes]
        """
        score_tensor_shape = score_tensor.shape
        label_mask_shape = label_mask.shape
        if score_tensor_shape != label_mask_shape:
            print('size not match')
            return
        
        scores = np.reshape(score_tensor, (score_tensor_shape[0] * score_tensor_shape[1], score_tensor_shape[2]))
        labels = np.reshape(label_mask, (label_mask_shape[0] * label_mask_shape[1], label_mask_shape[2]))

        for i in range(scores.shape[0]):
            score_idx = np.argmax(scores[i])
            self.preds.append(score_idx + 1)
            mask_idx = np.argmax(labels[i])
            self.labels.append(mask_idx + 1)
        
        for pred, label in zip(self.preds, self.labels):
            if pred != 1:
                print('test catch', pred, label)
    
    def calc_f1(self):
        # f1_score = 2 * (precision * recall) / (precision + recall)
        print('evaluate length: {}'.format(len(self.labels)))
        f1 = f1_score(y_true=self.labels, y_pred=self.preds, average='weighted')
        print('f1 score: {}'.format(f1))
        return f1
    
    def calc_accuracy(self):
        acc = [1 if label == pred else 0 for label, pred in zip(self.labels, self.preds)]
        return float(sum(acc)) / float(len(self.labels))