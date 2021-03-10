import json
import os

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt

import sys


speech_dir = "us_presidential_speeches/"


def president_terms(corpus, vocab=None):
    """
    Возврат tf-ids score для слов из словаря на переданном в качестве параметра текстовом корпусе
    :param corpus: корпус
    :param vocab: словарь
    :return: tf-ids
    """
    # создание Pipeline для преобразований, применение fit на переданном в качестве параметра корпусе
    pipe = Pipeline([('count', CountVectorizer(vocabulary=vocab)),
                     ('tfid', TfidfTransformer())]).fit(corpus)

    return pipe['tfid'].idf_


def get_data():
    if sys.argv[1] != '--terms':
        print('Incorrect arguments')
        sys.exit(1)

    try:
        i_title = sys.argv.index('--title')
        terms = sys.argv[2:i_title]
        title = sys.argv[i_title + 1]
    except:
        print('No title is given')
        terms = sys.argv[2:]
        title = ""

    return terms, title


def main():
    terms, title = get_data()
    get_score(terms, title)


def get_score(terms, title):
    # чтение набора JSON файлов в словарь
    # дата - ключ, текст речи - значение
    txts = {}
    for file in os.listdir(speech_dir):
        with open(speech_dir + file, "r") as infile:
            speech = json.load(infile)
            txts.update({speech["Date"]: speech["Speech"]})

    # создание датафрейма из словаря
    df = pd.DataFrame({'Date': list(txts.keys()), 'Speech': list(txts.values())})

    # преобразование столбца дат в формат datetime
    df.Date = pd.to_datetime(df.Date)

    # получение года из даты формата datetime
    df['Year'] = df.Date.dt.year

    # удаление колонки дата, т.к. на итоговом графике группировка по годам
    df.drop(columns=['Date'], inplace=True)

    # группировка датафрейма по годам. В поле Speech записываем список речей - текстовый корпус соответствующего года
    # получены две колонки: Year - год, Speech - list всех речей данного года
    df_years = df.groupby('Year').aggregate({'Speech': lambda x: [item for item in chain(x)]})

    # применяем написанную функцию получения tf-ids score к каждой строке датафрейма
    # для применения apply параметр vocab переведен в kwargs
    df_years['Score'] = df_years.Speech.apply(president_terms, vocab=terms)

    # разделение кортежей score на отдельные колонки
    # создаем датафрейм, в котором scores разделены по двум колонкам
    score_frame = pd.DataFrame(np.array(df_years.Score.values.tolist()),
                               columns=terms,
                               index=df_years.index)

    # объединяем полученный датафрейм score с исходным фреймом
    df_years = df_years.join(score_frame)

    # удаляем колонку с текстами речей, т.к. она не нужна
    df_years.drop(columns=['Speech'], inplace=True)

    # для использования pandas.plot создаем колонку Year из индекса
    df_years['Year'] = df_years.index

    # генерируем график встроенными средствами pandas
    # график отображается на канве matplotlib по умолчанию
    df_years.plot.line(x='Year', y=terms)

    # отображаем фигуру matplotlib
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    main()


