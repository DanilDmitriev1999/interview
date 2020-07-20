import glob, os
import numpy as np
import pandas as pd

def predict(path, model):
    result = {}
    os.chdir(path)
    for file in glob.glob("*.txt"):
        with open(file) as f:
            lines = f.readlines()
            predict = [predict_once(model, line.strip()) for line in lines]
            predict = 1 if np.mean(predict) > 0 else 0
            result.update({file: predict})

    file_name = [key for key, _ in result.items()]
    file_result = [value for _, value in result.items()]

    dataframe = pd.DataFrame({'filename': file_name,
                              'answer': file_result})

    dataframe.to_csv('prediction.csv')


if __name__ == '__main__':
    pass
