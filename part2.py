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
from _collections import OrderedDict

import re

# значение директорий по умолчанию
default_in_path = "us_presidential_speeches"
default_out_path = 'output_std'


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
    """
    Получение параметров из командной строки и запись этих параметров в словарь
    :return: словарь с параметрами
    """
    pos = {}
    values = {}
    args = ['--terms', '--path', '--title', '--output']

    # получение позиций названий аргументов типа --title
    for item in args:
        if item in sys.argv:
            pos.update({item[2:]: sys.argv.index(item)})
    pos = OrderedDict({k: v for k, v in sorted(pos.items(), key=lambda item: item[1])})

    # получение значений аргументов с использованием полученных позиций
    for i in range(len(list(pos.keys())) - 1):
        values.update({list(pos.keys())[i]: sys.argv[pos[list(pos.keys())[i]] + 1: pos[list(pos.keys())[i + 1]]]})
    values.update({list(pos.keys())[i + 1]: sys.argv[pos[list(pos.keys())[i + 1]] + 1:]})

    # проверка значений, подстановка умолчаний
    if 'terms' not in values.keys():
        print('No terms defined')
        sys.exit(1)

    if 'path' not in values.keys():
        values.update({'path': default_in_path})
    else:
        values['path'] = values['path'][0]

    if 'output' not in values.keys():
        filename = '_'.join(values['terms']) + '.png'
        filename = filename.replace(' ', '_')
        values.update({'output': default_out_path + '/' + filename})
    else:
        values['output'] = values['output'][0]

    if 'title' not in values.keys():
        values.update({'title': None})
    else:
        values['title'] = values['title'][0]

    return values


def main():
    """
    Основной метод программы
    Вызывается, если скрипт запущен как основной модуль
    :return:
    """
    get_score(get_data())


def get_score(values):
    """
    Метод получений корпуса текстов и JSON
    :param values: слловарь значений параметров
    :return:
    """
    # чтение набора JSON файлов в словарь
    # дата - ключ, текст речи - значение
    speech_dir = values['path']
    txts = {}
    try:
        for file in os.listdir(speech_dir):
            with open(speech_dir + '/' + file, "r") as infile:
                speech = json.load(infile)
                txts.update({speech["Date"]: speech["Speech"]})
    except IOError:
        sys.exit(1)

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
    df_years['Score'] = df_years.Speech.apply(president_terms, vocab=values['terms'])

    # разделение кортежей score на отдельные колонки
    # создаем датафрейм, в котором scores разделены по двум колонкам
    score_frame = pd.DataFrame(np.array(df_years.Score.values.tolist()),
                               columns=values['terms'],
                               index=df_years.index)

    # объединяем полученный датафрейм score с исходным фреймом
    df_years = df_years.join(score_frame)

    # удаляем колонку с текстами речей, т.к. она не нужна
    df_years.drop(columns=['Speech'], inplace=True)

    # для использования pandas.plot создаем колонку Year из индекса
    df_years['Year'] = df_years.index

    # генерируем график встроенными средствами pandas
    # график отображается на канве matplotlib по умолчанию
    df_years.plot.line(x='Year', y=values['terms'])

    # добавление заголовка, если он установлен
    if values['title']:
        plt.title(values['title'])

    # определение названий директории и файла из параметра output
    # запись графического изображения в файл с использование matplotlib
    try:
        path = re.search(r'[^\/]+$', values['output'])
        path = values['output'].replace(path.group(0), '')
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(values['output'])
    except IOError:
        print('Image save error')

    # отображаем фигуру matplotlib
    plt.show()


if __name__ == '__main__':
    main()


