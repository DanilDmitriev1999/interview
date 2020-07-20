import re


def preprocess_text(text):
    """
    Первоначальная обработка текста
    """
    text = text.lower().replace("  ", " ")
    text = text.replace('\xa0', ' ')
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL - Uniform Resource Locator', text)
    text = re.sub('[^\w\s]+|[\d]+', '', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


def predict_once(model, text):
    """
    Функиця предсказания языка по предложению
    """
    text = preprocess_text(text)
    predict = model.predict([text])
    return predict


if __name__ == '__main__':
    pass
