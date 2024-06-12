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


class Purifier():
    def openBinFile(self, _path):
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
        pass

    def zeroDeletion(self, _ind_array):
        """
        Функция удаления постоянной состоявляющей у векторов трейсов светорассеяния. Удаление происходит производится следующим образом:
        1. Определяется медиана вектора трейса
        2. Выбирается подвектор элементов вектора трейса, значение которых меньше медианы 
        3. Из вектора трейса вычитается среднее значение подвектора 

        Parameters
        ----------
        _ind_array : numpy.ndarray
            2D массив, набор векторов трейсов для которых производится удаление постоянной составляющей
        
        Returns
        ------
        _ind_array : numpy.ndarray
            2D массив, набор векторов трейсов c удаленной постоянной составляющей
        """
        N, M = np.shape(_ind_array)
        _ind_array_copy = np.copy(_ind_array)

        for i in range(N):
            _base = _ind_array_copy[i]
            _base = _base[_base < np.median(_base)]
            _ind_array_copy[i] = _ind_array_copy[i] - np.mean(_base)
        return _ind_array_copy
        pass

    def multiParticleRegistration(self):
        pass

    def quantileFilter(self, _features, _indexes, _quantile = 0.99):
        """
        Функция фильтрации набора векторов по доверительному интервалу близкому к 0.9. 
        Для обеспечения корректного сопоставления данных до и после фильтрации функция производит сохрание соответствующих индексов оригинального набора векторов.

        Parameters
        ----------
        _features : numpy.ndarray
            2D массив, набор векторов для которых производится вычисление центра масс
        _indexes : numpy.ndarray
            1D массив, индексы набора векторов

        Returns
        -------
        temp_features : numpy.ndarray
            Numpy 2D массив отфильтрованного набора векторов
        temp_indexes : numpy.ndarray
            Numpy 1D массив индексов отфильтрованного набора
        """
        edges_down = np.quantile(_features, (1 - _quantile) / 2 , axis=0)
        edges_up = np.quantile(_features, 1 - (1 - _quantile) / 2, axis=0)
        temp_indexes = _indexes
        temp_features = _features
        for i in range(len(edges_down)):
            temp_indexes = temp_indexes[temp_features[:,i] > edges_down[i]]
            temp_features = temp_features[temp_features[:,i] > edges_down[i]]
            temp_indexes = temp_indexes[temp_features[:,i] < edges_up[i]]
            temp_features = temp_features[temp_features[:,i] < edges_up[i]]
        return temp_features, temp_indexes
        
        pass

class Drawer():
    def drawMap(self, _map):
        plt.plot(_map[:,0], _map[:,1], '.')
        plt.show()
        pass

    def drawGating(self, _map, _map_names, _labels):

        plt.title('Результат кластеризации')
        plt.xlabel(_map_names[0])
        plt.ylabel(_map_names[1])

        for i in np.unique(_labels):
            _sub_map = _map[_labels == i]
            plt.plot(_sub_map[:,0], _sub_map[:,1], '.', label = "Кластер №" + str(i))

        plt.legend()
        plt.show()
        pass
    
    def drawIdnTraces(self, _labels, _forward_array, _backward_array):
        
        for i in np.unique(_labels):
            plt.figure(figsize=(10,5))
            plt.subplot(121)
            plt.title('Трейсы прямой индикатрисы кластера№' + str(i))
            plt.plot(_forward_array[_labels == i].T)
            plt.subplot(122)
            plt.title('Трейсы обратной индикатрисы кластера№' +  + str(i))
            plt.plot(_backward_array[_indexes_labels == i].T)
            plt.show()
            pass
		
class PltIdentifier():
    def identifyByMetric(self, _forward_array, _backward_array):

        forward_array = _forward_array / np.reshape(np.sum(_forward_array, axis = 1), (-1,1))
        backward_array = _backward_array / np.reshape(np.sum(_forward_array, axis = 1), (-1,1))

        _norm = self.__metric(forward_array, backward_array)
        _labels = AgglomerativeClustering(affinity="precomputed", linkage='complete', n_clusters=2).fit_predict(_norm)
        labels = np.copy(_labels)

        if np.mean(backward_array[_labels == 1]) < np.mean(backward_array[_labels == 0]):
            labels[_labels == 0] = 1
            labels[_labels == 1] = 0
        return labels
        pass

    def identifyByMap(self, _forward_array, _backward_array):
        pass

    def __metric(self, _forward_array, _backward_array):
        return distance_matrix(_backward_array, _backward_array) / (distance_matrix(_forward_array, _forward_array) + 0.0000000000001)
        pass

class AutoGater():
    def makeMap(self):
        pass
    
    def makeAutoGating(self):
        pass

    def __cmMap(self):
        pass

    def __integralMap(self):
        pass

    def __multifactorMap(self):
        pass

class InverseSolver():
    def __init__(self):
        self.xDB = None
        self.yDB = None
        self.KDTreeModel = None
        self.BallTreeModel = None


        pass
    def KDTreeFit(self):
        pass

    def BallTreeFit(self):
        pass
    
    def BruteforceFit(self):
        pass
    
    def MieSphereFit(self):
        pass

    def __VLFit(self):
        pass

    def __BayessianErrorEstimation(self):
        pass

    def __MieExtremaDetection(self):
        pass


