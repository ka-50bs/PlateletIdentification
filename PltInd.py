import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from LVBF import LV_fd

from scipy.spatial import distance_matrix
from tqdm import tqdm
from scipy.signal import medfilt
from scipy.signal import decimate
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
from scipy.optimize import direct
from sklearn.metrics import calinski_harabasz_score
from sklearn.decomposition import PCA


class PltIdn(object):
    def __init__(self):
        pass
        
    def init_data(self, _path):
        """
        Функция открытия файла данных СПЦ 3D_pockets_uint16.bin  и его конвертация в 3D numpy.ndarray 

        Parameters
        ----------
        _path : string 
            Путь до файла 3D_pockets_uint16.bin

        Returns
        -------
        data : numpy.ndarray
            Numpy 3D массив данных СПЦ.
        """
        reader = LV_fd(endian='>', encoding='cp1252')
        with open(_path) as reader.fobj:
            data = reader.read_array(reader.read_numeric, reader.LVuint16, ndims=3)
            data = np.array(data, dtype='float')
        return data
    
    def center_mass_map(self, _ind_array):
        """
        Функция вычисления карты центра масс для набора векторов (Cx, Cy).
        Применяется для разделения частиц по производной характеристике трейсов светорассеяния путем кластеризации. 

        Parameters
        ----------
        _ind_array : numpy.ndarray
            2D массив, набор векторов для которых производится вычисление центра масс

        Returns
        -------
        cm_map : numpy.ndarray
            Numpy 2D массив (карта) центров масс Cx, Cy
        """
        N, M = np.shape(_ind_array)
        time = np.array(range(M))
        cm_map = np.zeros((N, 2))
        for i in range(N):
            cm_map[i, 0] = np.sum(time * _ind_array[i,:]) / np.sum(_ind_array[i,:])
            cm_map[i, 1] = np.sum(time * _ind_array[i,:]) / np.sum(time)
        return cm_map
    
    def quantile_filter(self, features, indexes, quantile = 0.99):
        """
        Функция фильтрации набора векторов по доверительному интервалу близкому к 0.9. 
        Для обеспечения корректного сопоставления данных до и после фильтрации функция производит сохрание соответствующих индексов оригинального набора векторов.

        Parameters
        ----------
        features : numpy.ndarray
            2D массив, набор векторов для которых производится вычисление центра масс
        indexes : numpy.ndarray
            1D массив, индексы набора векторов

        Returns
        -------
        _features : numpy.ndarray
            Numpy 2D массив отфильтрованного набора векторов
        _indexes : numpy.ndarray
            Numpy 1D массив индексов отфильтрованного набора
        """
        edges_down = np.quantile(features, (1 - quantile) / 2 , axis=0)
        edges_up = np.quantile(features, 1 - (1 - quantile) / 2, axis=0)
        _indexes = indexes
        _features = features
        for i in range(len(edges_down)):
            _indexes = _indexes[_features[:,i] > edges_down[i]]
            _features = _features[_features[:,i] > edges_down[i]]
            _indexes = _indexes[_features[:,i] < edges_up[i]]
            _features = _features[_features[:,i] < edges_up[i]]
        return _features, _indexes
        
    def integral_map(self, forward_array, backward_array):
        """
        Функция вычисления карты интегралов для векторов переднего и заднего трейса.
        Применяется для разделения частиц по производной характеристике трейсов светорассеяния путем кластеризации.

        Parameters
        ----------
        forward_array : numpy.ndarray
            2D массив, набор векторов  трейсов прямой индикатрисы для которых производится вычисление интеграла
        backward_array : numpy.ndarray
            2D массив, набор векторов трейсов обратной индикатрисы для которых производится вычисление интеграла

        Returns
        -------
        int_map : numpy.ndarray
            Numpy 2D массив (карта) интегралов прямых и обратных трейсов
        """
        N, M = np.shape(forward_array)
        int_map = np.zeros((N, 2))
        for i in range(N):
            int_map[i,0] = np.sum(forward_array[i])
            int_map[i,1] = np.sum(backward_array[i])
        return int_map
    
    def zero_delection(self, _ind_array):
        """
        Функция удаления постоянной состоявляющей у векторов трейсов светорассеяния. Удаление происходит производится следующим образом:
        1. Определяется медиана вектора трейса
        2. Выбирается подвектор элементов вектора трейса, значение которых меньше медианы 
        3. Из вектора трейса вычитается среднее значение подвектора 

        Parameters
        ----------
        ind_array : numpy.ndarray
            2D массив, набор векторов трейсов для которых производится удаление постоянной составляющей
        
        Returns
        ------
        _ind_array : numpy.ndarray
            2D массив, набор векторов трейсов c удаленной постоянной составляющей
        """
        N, M = np.shape(ind_array)
        _ind_array_copy = np.copy(_ind_array)

        for i in range(N):
            _base = _ind_array_copy[i]
            _base = _base[_base < np.median(_base)]
            _ind_array_copy[i] = _ind_array_copy[i] - np.mean(_base)
        return _ind_array_copy
    
    def dim_reduction(self, _ind_array, win_size = 10):
        """
        Функция децимации входных векторов

        Parameters
        ----------
        _ind_array : numpy.ndarray
            2D массив, набор векторов трейсов для которых производится децимация
        
        Returns
        -------
        numpy.ndarray 2D массив, набор векторов децимированных трейсов 
        """
        return decimate(_ind_array, win_size, axis = 1)
        
    def auto_clustering_filter(self, forward_array, backward_array, indexes = None):
        """
        Функция автоматического разделения частиц на классы основанная на кластеризации 2D карты центра масс для
        трейсов переднего светорассения.
        Кластеризация происходит следующим образом:
        1. Вычисляются центры масс Сх и Су переднего трейса светорассеяния для каждой частицы
        2. Карты скалируются по методу МинМакс для обезразмеривания и происходит квантильная фильтрация частиц (99.9%)
        3. Происходит поиск оптимальных парамеров аффинной кластеразации методом DBSCAN. Поиск производится путем
        максимизации метрики оценки качества кластеризации (Calinski Harabasz Score) методом глобальной оптимизации
        DiRECT.
        4. На основе оптимальных параметров производится финальная кластеризация
        Для обеспечения корректного сопоставления данных до и после фильтрации функция производит сохрание соответствующих индексов оригинального набора векторов.
        
        Parameters
        ----------
        forward_array : numpy.ndarray
            2D массив, набор векторов трейсов прямой индикатрисы 
        backward_array : numpy.ndarray
            2D массив, набор векторов трейсов обратной индикатрисы
        indexes : numpy.ndarray
            1D массив, индексы набора векторов

        Returns
        -------
        _forward_array : numpy.ndarray 
            2D массив, набор векторов трейсов прямой индикатрисы, отфильтрованный квантильным методом 
        _backward_array : numpy.ndarray 
            2D массив, набор векторов трейсов обратной индикатрисы, отфильтрованный квантильным методом 
        _indexes : numpy.ndarray 
            1D массив, индексы набора векторов, отфильтрованный квантильным методом 
        _labels : numpy.ndarray 
            1D массив, метки кластеров для векторов прямой индикатрисы   
        """
        
        BOUNDS = [(0.001, 0.07), (10, 50)]
        _indexes = np.array(range(len(forward_array)))

        cm = self.center_mass_map(forward_array)
        Scaler = MinMaxScaler()
        cm = Scaler.fit_transform(cm)
        cm, _indexes = self.quantile_filter(cm, _indexes,0.999)

        def min_func(args): 
            labels = DBSCAN(eps=args[0], min_samples=int(args[1])).fit_predict(cm)
            if len(np.unique(labels)) == 1:
                return 1
            return 1 - calinski_harabasz_score(cm, labels)

        
        result = direct(min_func, BOUNDS, maxfun = 400)

        labels = DBSCAN(eps=result.x[0], min_samples=int(result.x[1])).fit_predict(cm)
        
        plt.title('Результат кластеризации по центру масс')
        plt.xlabel('CMX')
        plt.ylabel('CMY')
        
        for i in np.unique(labels):
            _cm = cm[labels == i]
            plt.plot(_cm[:,0],_cm[:,1], '.',label = 'Кластер№' + str(i),)
        
        plt.legend()
        plt.show()
        
        for i in np.unique(labels):
            plt.figure(figsize=(10,5))
            plt.subplot(121)
            plt.title('Трейсы прямой индикатрисы кластера№'+ str(i))
            plt.plot(forward_array[_indexes[labels == i]].T)
            plt.subplot(122)
            plt.title('Трейсы обратной  индикатрисы кластера№'+ str(i))
            plt.plot(backward_array[_indexes[labels == i]].T)
            plt.show()
            

        if type(indexes) != None:
            return forward_array[_indexes], backward_array[_indexes], indexes[_indexes], labels
        else:
            return forward_array[_indexes], backward_array[_indexes], _indexes, labels
    
    def dimer_idn(self, forward_array, backward_array):
        """
        Функция-алгоритм разделения частиц на классы отдельный тромбоцит \ агрегат по трейсам светорассеяния прямой и обратной индикатрис.
        Разделение происходит следующим образом:
        1. Производится нормировка векторов передней и задней индикатрисы на интеграл передней индикатрисы. Нормировка происходит для каждой индикатрисы индивидуально
        2. Строится матрица расстояний между частицами. Матрица расстояний определяется как отношение метрик L2 задней и передней интикатрисы.
        3. По известной матрице расстояний производится агломеративная кластеризация на 2 кластера.
        4. После кластеризации производится вычисление среднего значение интеграла отнормированной задней индикатрисы для каждого из выдленных кластеров.
        Тот кластер, чей средний интеграл больше - определяется как как кластер агрегатов.

        Parameters
        ----------
        forward_array : numpy.ndarray
            2D массив, набор векторов трейсов прямой индикатрисы 
        backward_array : numpy.ndarray
            2D массив, набор векторов трейсов обратной индикатрисы

        Returns
        -------
        labels : 1D массив, метки частиц, 0 соответсвует мономеру тромбоцитов, 1 соотсветствует агрегату
        """

        _indexes = np.array(range(len(forward_array)))

        def dimer_norm(_forward_array, _backward_array):
            return distance_matrix(_backward_array, _backward_array) / (distance_matrix(_forward_array, _forward_array) + 0.00000000001)

        forward_array = forward_array / np.reshape(np.sum(forward_array, axis = 1), (-1,1))
        backward_array = backward_array / np.reshape(np.sum(forward_array, axis = 1), (-1,1))

        _norm = dimer_norm(forward_array, backward_array)
        
        _labels = AgglomerativeClustering(affinity="precomputed", linkage='complete', n_clusters=2).fit_predict(_norm)

        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.title('Трейсы прямой индикатрисы кластера№0')
        plt.plot(forward_array[_indexes[_labels == 0]].T)
        plt.subplot(122)
        plt.title('Трейсы обратной индикатрисы кластера№0')
        plt.plot(backward_array[_indexes[_labels == 0]].T)
        plt.show()
        
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.title('Трейсы прямой индикатрисы кластера№1')
        plt.plot(forward_array[_indexes[_labels == 1]].T)
        plt.subplot(122)
        plt.title('Трейсы обратной индикатрисы кластера№1')
        plt.plot(backward_array[_indexes[_labels == 1]].T)
        plt.show()

        print('Размер кластера№0: ' + str(len(backward_array[_indexes[_labels == 0]])))
        print('Размер кластера№1: ' + str(len(backward_array[_indexes[_labels == 1]])))

        labels = np.copy(_labels)

        if np.mean(backward_array[_indexes[_labels == 1]]) < np.mean(backward_array[_indexes[_labels == 0]]):
            labels[_labels == 0] = 1
            labels[_labels == 1] = 0
        return labels
    
    def draw_cluster_map(self, cm, labels):
        pass

    def draw_platelet_ind(self):
        pass
    