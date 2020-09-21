import numpy as np

def get_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

if __name__ == '__main__':
    precision = 0.454
    recall = 0.349
    f1 = get_f1(precision, recall)
    print(f1)