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
        _features : numpy.ndarray
            Numpy 2D массив отфильтрованного набора векторов
        __indexes : numpy.ndarray
            Numpy 1D массив индексов отфильтрованного набора
        """
        edges_down = np.quantile(_features, (1 - _quantile) / 2 , axis=0)
        edges_up = np.quantile(_features, 1 - (1 - _quantile) / 2, axis=0)
        __indexes = _indexes
        _features = _features
        for i in range(len(edges_down)):
            __indexes = __indexes[_features[:,i] > edges_down[i]]
            _features = _features[_features[:,i] > edges_down[i]]
            __indexes = __indexes[_features[:,i] < edges_up[i]]
            _features = _features[_features[:,i] < edges_up[i]]
        return _features, __indexes
        
        pass

class Drawer():
    def drawMap(self):
        pass

    def drawGating(self):
        pass
    
    def drawIdentification(self):
        pass
		
class PltIdentifier():
    def identifyByMetric(self):
        pass

    def identifyByMap(self):
        pass

    def __metric(self):
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


