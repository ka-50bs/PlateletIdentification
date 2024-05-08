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



def divide_numbers(a, b):
    """
    Divide two numbers.
 
    Parameters
    ----------
    a : float
        The dividend.
    b : float
        The divisor.
 
    Returns
    -------
    float
        The quotient of the division.
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b

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
    
    def center_mass_map(self, ind_array):
        """
        Функция вычисления карты центра масс для набора векторов (Cx, Cy).
        Применяется для разделения частиц по производной характеристике трейсов светорассеяния путем кластеризации. 

        Parameters
        ----------
        ind_array : numpy.ndarray
            2D массив, набор векторов для которых производится вычисление центра масс

        Returns
        -------
        cm_map : numpy.ndarray
            Numpy 2D массив (карта) центров масс Cx, Cy
        """
        N, M = np.shape(ind_array)
        time = np.array(range(M))
        cm_map = np.zeros((N, 2))
        for i in range(N):
            cm_map[i, 0] = np.sum(time * ind_array[i,:]) / np.sum(ind_array[i,:])
            cm_map[i, 1] = np.sum(time * ind_array[i,:]) / np.sum(time)
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
            2D массив, набор векторов прямых трейсов для которых производится вычисление интеграла
        backward_array : numpy.ndarray
            2D массив, набор векторов обратных трейсов для которых производится вычисление интеграла

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
    
    def zero_delection(self, ind_array):
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
        _ind_array = np.copy(ind_array)

        for i in range(N):
            _base = _ind_array[i]
            _base = _base[_base < np.median(_base)]
            _ind_array[i] = _ind_array[i] - np.mean(_base)
        return _ind_array
    
    def dim_reduction(self, ind_array, win_size = 10):
        """
        Функция децимации входных векторов

        Parameters
        ----------
        ind_array : numpy.ndarray
            2D массив, набор векторов трейсов для которых производится децимация
        
        Returns
        -------
        numpy.ndarray 2D массив, набор векторов децимированных трейсов 
        """
        return decimate(ind_array, win_size, axis = 1)
        
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
            2D массив, набор векторов прямых трейсов для которых производится вычисление интеграла
        backward_array : numpy.ndarray
            2D массив, набор векторов обратных трейсов для которых производится вычисление интеграла
        indexes : numpy.ndarray
            1D массив, индексы набора векторов

        Returns
        -------

        """

        _indexes = np.array(range(len(forward_array)))

        cm = self.center_mass_map(forward_array)
        Scaler = MinMaxScaler()
        cm = Scaler.fit_transform(cm)
        cm, _indexes = self.quantile_filter(cm, _indexes,0.99)

        def min_func(args): 
            clustering = DBSCAN(eps=args[0], min_samples=int(args[1])).fit_predict(cm)
            if len(np.unique(clustering)) == 1:
                return 1
            return 1 - calinski_harabasz_score(cm, clustering)

        BOUNDS = [(0.001, 0.07), (10, 50)]
        result = direct(min_func, BOUNDS, maxfun = 400)

        clustering = DBSCAN(eps=result.x[0], min_samples=int(result.x[1])).fit_predict(cm)
        
        plt.title('Результат кластеризации по центру масс')
        plt.xlabel('CMX')
        plt.ylabel('CMY')
        
        for i in np.unique(clustering):
            _cm = cm[clustering == i]
            plt.plot(_cm[:,0],_cm[:,1], '.',label = 'Кластер№' + str(i),)
        
        plt.legend()
        plt.show()
        
        for i in np.unique(clustering):
            plt.figure(figsize=(10,5))
            plt.subplot(121)
            plt.title('Трейсы прямой индикатрисы кластера№'+ str(i))
            plt.plot(forward_array[_indexes[clustering == i]].T)
            plt.subplot(122)
            plt.title('Трейсы обратной  индикатрисы кластера№'+ str(i))
            plt.plot(backward_array[_indexes[clustering == i]].T)
            plt.show()
            
        choise = int(input('Какой выбрать кластер?'))
        _indexes = _indexes[clustering == choise]
        
        print('Количество выбранных частиц:' + str(len(_indexes)))
        if type(indexes) != None:
            return forward_array[_indexes], backward_array[_indexes], indexes[_indexes]
        else:
            return forward_array[_indexes], backward_array[_indexes], _indexes
    
    def dimer_idn(self, forward_array, backward_array, indexes = None):
        """

        """

        _indexes = np.array(range(len(forward_array)))

        def dimer_norm(_forward_array, _backward_array):
            return distance_matrix(_backward_array, _backward_array) / (distance_matrix(_forward_array, _forward_array) + 0.00000000001)
        
        _norm = dimer_norm(forward_array, backward_array)
        clustering = AgglomerativeClustering(affinity="precomputed", linkage='complete', n_clusters=2).fit_predict(_norm)

        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.title('Трейсы прямой индикатрисы кластера№0')
        plt.plot(forward_array[_indexes[clustering == 0]].T)
        plt.subplot(122)
        plt.title('Трейсы обратной индикатрисы кластера№0')
        plt.plot(backward_array[_indexes[clustering == 0]].T)
        plt.show()
        
        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.title('Трейсы прямой индикатрисы кластера№1')
        plt.plot(forward_array[_indexes[clustering == 1]].T)
        plt.subplot(122)
        plt.title('Трейсы обратной индикатрисы кластера№1')
        plt.plot(backward_array[_indexes[clustering == 1]].T)
        plt.show()

        print('Размер кластера№0: ' + str(len(backward_array[_indexes[clustering == 0]])))
        print('Размер кластера№1: ' + str(len(backward_array[_indexes[clustering == 1]])))

    def dimer_multifactor_idn(self, forward_array, backward_array, indexes = None):

        _indexes = np.array(range(len(forward_array)))
        
        int_map = self.integral_map(forward_array, backward_array)
        cmf_map = self.center_mass_map(forward_array)
        cmb_map = self.center_mass_map(backward_array)

        multifactor_map = np.hstack([int_map, cmf_map, cmb_map])

        pca = PCA()
        multifactor_map = pca.fit_transform(multifactor_map)

        Scaler = MinMaxScaler()
        multifactor_map = Scaler.fit_transform(multifactor_map)
        multifactor_map, _indexes = self.quantile_filter(multifactor_map, _indexes,0.99)
        multifactor_map = Scaler.fit_transform(multifactor_map)
        multifactor_map = multifactor_map[:,:2]
        
        def min_func(args): 
            clustering = DBSCAN(eps=args[0], min_samples=int(args[1])).fit_predict(multifactor_map)
            if len(np.unique(clustering)) == 1:
                return 1
            return 1 - calinski_harabasz_score(multifactor_map, clustering)

        BOUNDS = [(0.001, 0.07), (10, 50)]
        result = direct(min_func, BOUNDS, maxfun = 400)

        clustering = DBSCAN(eps=result.x[0], min_samples=int(result.x[1])).fit_predict(multifactor_map)
        

        plt.title('Результат кластеризации по центру масс')
        plt.xlabel('CMX')
        plt.ylabel('CMY')
        
        for i in np.unique(clustering):
            _multifactor_map = multifactor_map[clustering == i]
            plt.plot(_multifactor_map[:,0],_multifactor_map[:,1], '.',label = 'Кластер№' + str(i),)
        
        plt.legend()
        plt.show()

        for i in np.unique(clustering):
            plt.figure(figsize=(10,5))
            plt.subplot(121)
            plt.title('Трейсы прямой индикатрисы кластера№'+ str(i))
            plt.plot(forward_array[_indexes[clustering == i]].T)
            plt.subplot(122)
            plt.title('Трейсы обратной  индикатрисы кластера№'+ str(i))
            plt.plot(backward_array[_indexes[clustering == i]].T)
            plt.show()
            
        choise = int(input('Какой выбрать кластер?'))
        _indexes = _indexes[clustering == choise]
        
        print('Количество выбранных частиц:' + str(len(_indexes)))
        if type(indexes) != None:
            return forward_array[_indexes], backward_array[_indexes], indexes[_indexes]
        else:
            return forward_array[_indexes], backward_array[_indexes], _indexes



        # plt.figure(figsize=(10,5))
        # plt.subplot(121)
        # plt.title('Трейсы прямой индикатрисы кластера№0')
        # plt.plot(forward_array[_indexes[clustering == 0]].T)
        # plt.subplot(122)
        # plt.title('Трейсы обратной индикатрисы кластера№0')
        # plt.plot(backward_array[_indexes[clustering == 0]].T)
        # plt.show()
        
        # plt.figure(figsize=(10,5))
        # plt.subplot(121)
        # plt.title('Трейсы прямой индикатрисы кластера№1')
        # plt.plot(forward_array[_indexes[clustering == 1]].T)
        # plt.subplot(122)
        # plt.title('Трейсы обратной индикатрисы кластера№1')
        # plt.plot(backward_array[_indexes[clustering == 1]].T)
        # plt.show()

        # print('Размер кластера№0: ' + str(len(backward_array[_indexes[clustering == 0]])))
        # print('Размер кластера№1: ' + str(len(backward_array[_indexes[clustering == 1]])))