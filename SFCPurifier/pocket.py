"""
SFCPurifier
====

@author: ka-50bs
"""
import random
import miepython
import nmslib
from tqdm import tqdm
from scipy.spatial import distance_matrix
from tqdm import tqdm
from scipy.signal import medfilt
from scipy.signal import decimate
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
from scipy.optimize import direct
from sklearn.metrics import calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.interpolate import InterpolatedUnivariateSpline

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .LVBF import LV_fd
from scipy.fft import fft

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from scipy.signal import find_peaks

class Purifier():
    def openBinFile(self, path):
        """
        Функция открытия файла данных СПЦ 3D_pockets_uint16.bin  и его конвертация в 3D numpy.ndarray 

        Parameters
        ----------
        path : string 
            Путь до файла 3D_pockets_uint16.bin

        Returns
        -------
        data : numpy.ndarray
            Numpy 3D массив данных СПЦ.
        """
        reader = LV_fd(endian='>', encoding='cp1252')
        with open(path) as reader.fobj:
            data = reader.read_array(reader.read_numeric, reader.LVuint16, ndims=3)
            data = np.array(data, dtype='float')
        return data
        pass

    def openBigPocketBinFile(self, path, peak_height = 1000):
        """
        Функция открытия файла данных большоего объема СПЦ 3D_pockets_uint16.bin и его конвертация в 3D numpy.ndarray
        Parameters
        ----------
        path : string 
            Строка, путь до файла 3D_pockets_uint16.bin
        
        peak_height : int
            Число, пороговая высота триггерного пика которая позволяет идентифицировать частицу в большом пакете

        Returns
        -------
        data : numpy.ndarray
            Numpy 3D массив данных СПЦ.
        """
        data = self.openBinFile(path)
        pack = []
        for i in range(len(data)):
            peaks, _ = find_peaks(data[i,3,:] - np.mean(data[i,3,:]), height = peak_height, distance = 5000)
            for peak in peaks:
                if (peak - 1000) > 0 and (peak-95000) < 0:
                    pack.append(data[i,:, peak - 500 : 4500 + peak])
        data = np.array(pack)
        return data
      
    def zeroDeletion(self, trace_array):
        """
        Функция удаления постоянной состоявляющей у векторов трейсов светорассеяния. Удаление происходит производится следующим образом:
        1. Определяется медиана вектора трейса
        2. Выбирается подвектор элементов вектора трейса, значение которых меньше медианы 
        3. Из вектора трейса вычитается среднее значение подвектора 

        Parameters
        ----------
        trace_array : numpy.ndarray
            2D массив, набор векторов трейсов для которых производится удаление постоянной составляющей
        
        Returns
        ------
        trace_array_copy : numpy.ndarray
            2D массив, набор векторов трейсов c удаленной постоянной составляющей
        """
        N, M = np.shape(trace_array)
        trace_array_copy = np.copy(trace_array)

        for i in range(N):
            base = trace_array_copy[i]
            base = base[base < np.quantile(base, 0.50, method='closest_observation')]
            if len(base)!=0:
                trace_array_copy[i] = trace_array_copy[i] - np.mean(base)

        return trace_array_copy
        pass

    def quantileFilter(self, features, indexes, _quantile = 0.99):
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
        temp_features : numpy.ndarray
            Numpy 2D массив отфильтрованного набора векторов
        temp_indexes : numpy.ndarray
            Numpy 1D массив индексов отфильтрованного набора
        """
        edges_down = np.quantile(features, (1 - _quantile) / 2 , axis=0)
        edges_up = np.quantile(features, 1 - (1 - _quantile) / 2, axis=0)
        temp_indexes = np.copy(indexes)
        temp_features = np.copy(features)

        for i in range(len(edges_down)):
            temp_indexes = temp_indexes[temp_features[:,i] > edges_down[i]]
            temp_features = temp_features[temp_features[:,i] > edges_down[i]]
            temp_indexes = temp_indexes[temp_features[:,i] < edges_up[i]]
            temp_features = temp_features[temp_features[:,i] < edges_up[i]]
        return temp_features, temp_indexes
        pass

class Drawer():
    def drawMap(self, features_map):
        """
        Функция отрисовки 2D карты параметров.
        
        Parameters
        ----------
        features_map : numpy.ndarray
            Numpy 2D массив отрисовываемых параметров.
        """
        plt.plot(features_map[:,0], features_map[:,1], '.')
        plt.xlim([-0.5,1.5])
        plt.ylim([-0.5,1.5])
        plt.show()
        pass

    def drawGating(self, features_map, map_names, labels, title = ''):
        """
        Функция отрисовки 2D карты параметров с учетом меток кластерицации.
        
        Parameters
        ----------
        features_map : numpy.ndarray
            Numpy 2D массив отрисовываемых параметров.
        map_names : list [str]
            Список строк наименований параметров отрисовываемой 2D карты
        labels : numpy.ndarray
            Numpy 1D массив меток кластеров для каждого элемента массива features_map
        title : str
            Строка-заголовок отрисовываемой 2D карты
        """
        plt.title('Результат кластеризации: ' + title)
        plt.xlabel(map_names[0])
        plt.ylabel(map_names[1])

        for i in np.unique(labels):
            sub_map = features_map[labels == i]
            plt.plot(sub_map[:,0], sub_map[:,1], '.', label = "Кластер №" + str(i) + ', размер кластера: ' + str(len(sub_map)))
        plt.xlim([-0.2,1.2])
        plt.ylim([-0.2,1.2])
        plt.legend()
        plt.show()
        pass
    def drawSpheresParamsStat(self, params_array, params_names):
        for i in range(len(params_names)):
            plt.figure(figsize=(10,5))
            plt.suptitle('Размер кластера: ' + str(len(forward_trace_array[labels == i])))
            plt.subplot(21)
            plt.plot

    def drawIdnTraces(self, forward_trace_array, backward_trace_array, labels = None):
        """
        Функция отрисовки трейсов с учетом метки кластеризации.
        Для каждой уникальной метки производится отрисовка соответствующих трейсов переднего и заднего рассеяния.
        
        Parameters
        ----------        
        forward_trace_array : numpy.ndarray
            Numpy 2D массив трейсов переднего рассеяния
        backward_trace_array : numpy.ndarray
            Numpy 2D массив трейсов заднего рассеяния
        labels : numpy.ndarray
            Numpy 1D массив меток кластеров для каждого элемента массивов forward_trace_array и backward_trace_array
             
        """
        if np.all(labels) != None:
            for i in np.unique(labels):
                plt.figure(figsize=(10,5))
                plt.suptitle('Размер кластера: ' + str(len(forward_trace_array[labels == i])))
                plt.subplot(121)
                plt.title('Трейсы прямой индикатрисы кластера№' + str(i))
                plt.plot(forward_trace_array[labels == i].T)
                plt.subplot(122)
                plt.title('Трейсы обратной индикатрисы кластера№'  + str(i))
                plt.plot(backward_trace_array[labels == i].T)
                plt.show()
        else:
            plt.figure(figsize=(10,5))
            plt.subplot(121)
            plt.title('Трейсы прямой индикатрисы')
            plt.plot(forward_trace_array.T)
            plt.subplot(122)
            plt.title('Трейсы обратной индикатрисы')
            plt.plot(backward_trace_array.T)
            plt.show() 
		
class PltIdentifier():
    def identifyByMetric(self, forward_array, backward_array):
        """
        Функция решения задачи идентификации одиночный тромбоцит\агрегат с использованием агломеративной кластеризации по предварительно 
        инициализированной матрицей попарных расстояний между частицами. Попарное расстояние является функцией трейсов прямого и обратного рассеяния.
        
        Parameters
        ----------
        forward_trace_array : numpy.ndarray
            Numpy 2D массив трейсов переднего рассеяния
        backward_trace_array : numpy.ndarray
            Numpy 2D массив трейсов заднего рассеяния  

        Returns
        ------- 
        _labels : numpy.ndarray
            Numpy 1D массив меток идентификации частиц по трейсам светорассеяния 

        """
        forw = forward_array / np.reshape(np.sum(forward_array, axis = 1), (-1,1))
        back = backward_array / np.reshape(np.sum(forward_array, axis = 1), (-1,1))

        norm = self.__metric(forw, back)
        labels = AgglomerativeClustering(affinity="precomputed", linkage='complete', n_clusters=2).fit_predict(norm)
        _labels = np.copy(labels)

        if np.mean(back[labels == 1]) < np.mean(back[labels == 0]):
            _labels[labels == 0] = 1
            _labels[labels == 1] = 0
        return _labels
        pass

    def identifyByMap(self, forward_array, backward_array):
        pass

    def __metric(self, forward_array, backward_array):
        """
        Функция метрики попарных расстояний между частицами. Аргументы функции - трейсы переднего и заднего рассеяния.
        
        Parameters
        ----------
        forward_trace_array : numpy.ndarray
            Numpy 2D массив трейсов переднего рассеяния
        backward_trace_array : numpy.ndarray
            Numpy 2D массив трейсов заднего рассеяния
        
        Returns
        -------
            Numpy 2D массив, матрица попарных расстояний 
        """
        forw = decimate(forward_array, 10, axis = 1)
        back = decimate(backward_array, 10, axis = 1)
        return distance_matrix(back, back) / (distance_matrix(forw, forw) + 0.0000000000001)
        pass

class AutoGater():
    def makeMap(self, forward_array, backward_array, map_type = 'CM', norm_type = 'MinMax'):
        """
        Функция генерации карты параметров трейсов светорассеяния. Позволяет вычислять карты центра масс,
        карты интегралов и мультифакторной карты 

        Parameters
        ----------
        forward_array : numpy.ndarray
            2D массив, набор векторов для которых производится вычисление интеграла
        backward_array : numpy.ndarray
            2D массив, набор векторов для которых производится вычисление интеграла
        map_type : str
            Строка определяющая тип требуемой для построения карты. 
            Доступны 3 варианта карт:
                "CM" : центр масс
                "integral" : интеграл переднего и заднего рассеяния
                "multifactor" : мультифакторная карта
        
        norm_type : str
            Строка определяющая тип нормализатора карты
            Доступны 3 варианта нормализатора:
                "Quantile" : квантильный 
                "MinMax" : мин\макс 
                "Native" : без нормализатора
        """

        gened_map = None
        if map_type == 'CM':
            gened_map = self.__cmMap(forward_array)
        elif map_type == 'integral':
            gened_map = self.__integralMap(forward_array, backward_array)
        elif map_type == 'multifactor':
            gened_map = self.__multifactorMap(forward_array, backward_array)
        elif map_type == 'fft':
            gened_map = self.__fftMap(forward_array, backward_array)
        
        scaler = None
        if norm_type == 'Quantile':
            scaler = QuantileTransformer(output_distribution='normal')
            gened_map = scaler.fit_transform(gened_map)
            scaler = MinMaxScaler()
            gened_map = scaler.fit_transform(gened_map)

        elif norm_type == 'MinMax':
            scaler = MinMaxScaler()
            gened_map = scaler.fit_transform(gened_map)

        elif norm_type == 'Native':
            None
        else:
            gened_map = norm_type.transform(gened_map)

        return gened_map, scaler
        pass
    
    def makeAutoGating(self, gened_map, BOUNDS = [(0.001, 0.1), (2, 200)]):
        """
        Функция автоматического разделения частиц на классы основанная на кластеризации карты параметров,
        рассчитанных на основе производных параметров частиц.

        Кластеризация происходит следующим образом:
        1. Происходит поиск оптимальных парамеров аффинной кластеразации методом DBSCAN. Поиск производится путем
        максимизации метрики оценки качества кластеризации (Calinski Harabasz Score) методом глобальной оптимизации
        DiRECT.
        2. На основе оптимальных параметров производится финальная кластеризация
        
        Parameters
        ----------
        gened_map : numpy.ndarray
            2D массив, карта кластеризуемых параметров
        Returns
        -------
        labels : numpy.ndarray 
            1D массив, метки кластеров для карты параметров
        """
        
        # BOUNDS = [(0.001, 0.05), (10, 30)]
        # BOUNDS = [(0.001, 0.1), (2, 200)]

        def min_func(args): 
            labels = DBSCAN(eps=args[0], min_samples=int(args[1])).fit_predict(gened_map)
            # labels = OPTICS(max_eps=args[0], min_samples=int(args[1])).fit_predict(gened_map)
            if len(np.unique(labels)) == 1:
                return 100000000000000
            return -calinski_harabasz_score(gened_map, labels)
        
        result = direct(min_func, BOUNDS, maxfun = 100)
        print(result.x)
        # labels = OPTICS(max_eps=result.x[0], min_samples=int(result.x[1])).fit_predict(gened_map)
        labels = DBSCAN(eps=result.x[0], min_samples=int(result.x[1])).fit_predict(gened_map)
        return labels
        pass

    def __cmMap(self, trace_array):
        """
        Функция вычисления карты центра масс для набора векторов трейсов или индикатрис (Cx, Cy).
        Применяется для разделения частиц по производной характеристике трейсов или индикатрис светорассеяния путем кластеризации. 

        Parameters
        ----------
        trace_array : numpy.ndarray
            2D массив, набор векторов для которых производится вычисление центра масс

        Returns
        -------
        cm_map : numpy.ndarray
            Numpy 2D массив (карта) центров масс Cx, Cy
        """
        N, M = np.shape(trace_array)
        time = np.array(range(M))
        cm_map = np.zeros((N, 2))
        for i in range(N):
            cm_map[i, 0] = np.sum(time *trace_array[i,:]) / np.sum(trace_array[i,:])
            cm_map[i, 1] = np.sum(time * trace_array[i,:]) / np.sum(time)
        return cm_map

        pass

    def __integralMap(self, forward_array, backward_array):
        """
        Функция вычисления карты ингегралов светорассеяния для набора векторов трейсов или индикатрис прямого и обратного рассеяния.
        Применяется для разделения частиц по производной характеристике трейсов светорассеяния путем кластеризации.

        Parameters
        ----------
        forward_array : numpy.ndarray
            2D массив, набор векторов для которых производится вычисление интеграла
        backward_array : numpy.ndarray
            2D массив, набор векторов для которых производится вычисление интеграла

        Returns
        -------
        int_map : numpy.ndarray
            Numpy 2D массив (карта) интегралов
        """

        N, M = np.shape(forward_array)
        int_map = np.zeros((N, 2), dtype='float')
        for i in range(N):
            int_map[i, 0] = np.sum(forward_array[i])
            int_map[i, 1] = np.sum(backward_array[i])
        # return np.log(int_map + 0.1)
        return int_map
        pass

    def __fftMap(self, forward_array, backward_array):
        """
        Функция вычисления карты ингегралов светорассеяния для набора векторов трейсов или индикатрис прямого и обратного рассеяния.
        Применяется для разделения частиц по производной характеристике трейсов светорассеяния путем кластеризации.

        Parameters
        ----------
        forward_array : numpy.ndarray
            2D массив, набор векторов для которых производится вычисление интеграла
        backward_array : numpy.ndarray
            2D массив, набор векторов для которых производится вычисление интеграла

        Returns
        -------
        int_map : numpy.ndarray
            Numpy 2D массив (карта) интегралов
        """

        N, M = np.shape(forward_array)
        fft_map = np.zeros((N, 2), dtype='float')
        for i in range(N):

            x = np.abs(fft(forward_array[i]))
            y = np.abs(fft(backward_array[i]))
            
            x_peaks = find_peaks(x, height=np.max(x) / 7)
            y_peaks = find_peaks(y, height=np.max(y) / 7)
            if len(x_peaks[0]) > 0:
                fft_map[i, 0] = x_peaks[0][0]
            else:
                fft_map[i, 0] = 0

            if len(y_peaks[0]) > 0:
                fft_map[i, 1] = y_peaks[0][0]
            else:
                fft_map[i, 1] = 0

        return fft_map
        pass

    def __multifactorMap(self, forward_array, backward_array):
        return np.hstack([self.__cmMap(forward_array), self.__integralMap(forward_array, backward_array)])
        pass

class InverseSolver():
    def __init__(self):
        self.x_DB = None
        self.y_DB = None
        self.NearestNeighborsModels = None
        self.NearestNeighborsRegressorModels = None
        pass

    def weightM(self, θ):
        f_x = []
        for i in range(len(θ)):
            if θ[i] == 0:
                f_x.append(0)
            elif θ[i] < 90:
                f_x.append((1 / θ[i]) * np.exp(-2 * (np.log(θ[i] / 54)) ** 2))
            elif θ[i] >= 90:
                f_x.append((1 / (180 - θ[i])) * np.exp(-2 * (np.log((180 - θ[i]) / 54)) ** 2))
        return np.array(f_x)

    def initModels(self, x_DB, y_DB): 
        """
        Функция инициализации моделей поиска ближайших соседей. 
        """
        self.x_DB = np.copy(x_DB)
        self.y_DB = np.copy(y_DB)
        self.NearestNeighborsModels = {
            'KDTree': NearestNeighbors(algorithm = 'kd_tree', leaf_size = 20).fit(x_DB),
            'BallTree': NearestNeighbors(algorithm = 'ball_tree', leaf_size = 20).fit(x_DB),
            'BruteForce': NearestNeighbors(algorithm = 'brute').fit(x_DB) 
            }

        self.NearestNeighborsRegressorModels = {
            'KDTree': KNeighborsRegressor(n_neighbors = 10, algorithm = 'kd_tree', leaf_size = 20).fit(x_DB, y_DB),
            'BallTree': KNeighborsRegressor(n_neighbors = 10, algorithm = 'ball_tree', leaf_size = 20).fit(x_DB, y_DB),
            'BruteForce': KNeighborsRegressor(n_neighbors = 10, algorithm = 'brute').fit(x_DB, y_DB) 
            }
        pass
    
    def NearestNeighborsFit(self, x_Exp, neighbors = 1000, model = 'KDTree'):
        """
        Функция решения обратной задачи светорассеяния путем подгонки теоретического сигнала к экспериментальному методом поиска ближайщего 
        соседа в базе данных.

        Parameters
        ----------
        
        """
        res_DB, idx_DB = self.NearestNeighborsModels[model].kneighbors(x_Exp, neighbors)
        return res_DB, idx_DB

    def NearestNeighborsRegressorFit(self,  x_Exp, neighbors = 1000, model = 'KDTree'):
        res_DB, idx_DB = self.NearestNeighborsRegressorModels[model].kneighbors(x_Exp, neighbors)
        y_DB = self.y_DB[idx_DB]
        z = x_Exp - self.x_DB[0]
        return self.NearestNeighborsRegressorModels[model].predict(x_Exp)
        pass


        pass

    def __BayessianEstimation(self, y_BD, res_BD, z):

        def autocorr(x):
            N=len(x)
            corr = []
            for i in range(N):
                summ = []
                for j in range(N-i):
                    summ.append(x[i+j] * x[j])
                corr.append(np.sum(summ) / np.sum(x ** 2))
            return np.array(corr)
        
        def FreedomDegree(x):
                pk = autocorr(x)
                N = len(x)
                sumi = 0

                for k in range(1,N-1):
                    sumi = sumi + (N - k) * pk[k] ** 2

                n = (N ** 2) / (N + 2 * sumi) 
                return n

        
        n = FreedomDegree(z)
        S = res_BD ** (-n / 2)
        S = S / np.sum(S)
        S = np.reshape(S, (-1,1))
        mean = np.sum(S * y_BD, axis = 0)
        cov = 0
        for i in  range (len(S)):
            vector = np.reshape((y_BD[i] - mean), (1, -1))
            cov = cov + S[i] * vector * vector.T
            
        return mean, np.diagonal(cov) ** 0.5
        pass
    
    def BayessianEstimation(self, x_Exp, solver = 'DBSearch', neighbors = 1000):
        dist, idx  = self.NearestNeighborsFit(x_Exp, neighbors=neighbors, model='BruteForce')
        means, sigmas = [], []
        for i in tqdm(range(len(x_Exp))):
             
            bayes = self.__BayessianEstimation(self.y_DB[idx[i]], 
                                                dist[i], 
                                                x_Exp[i] - self.x_DB[idx[i][0]])
            means.append(bayes[0])
            sigmas.append(bayes[1])
        return np.array(means), np.array(sigmas)
        pass

class InverseSpheresSolver():
    def __init__(self):
        self.x_DB = None
        self.y_DB = None
        self.KNNModel = None
        self.angles = None
        self.mu = None

    def weightM(self, θ):
        """
        Весовая функция, используемая в решении обратной задачи светорассеяния для индикатрис светорассеяния.

        Parameters
        ----------
        θ : numpy.ndarray
            1D массив, набор углов в градусах, область определения весовой функции 
        Returns
        -------
        mf : numpy.ndarray
            1D массив весовой функции
        """

        f_x = []
        for i in range(len(θ)):
            if θ[i] == 0:
                f_x.append(0)
            elif θ[i] < 90:
                f_x.append((1 / θ[i]) * np.exp(-2 * (np.log(θ[i] / 54)) ** 2))
            elif θ[i] >= 90:
                f_x.append((1 / (180 - θ[i])) * np.exp(-2 * (np.log((180 - θ[i]) / 54)) ** 2))
        mf = np.array(f_x)
        return mf

    def initParams(self, lambdaForw, angles, n_env = 1.33333, rf_path = 'tf.txt'):
        """
        Метод инициализации параметров СПЦ. Для инициализации требуется несколько параметров 
        """
        self.angles = angles
        self.mu = np.cos(np.deg2rad(angles))
        self.lambdaForw = lambdaForw
        self.lambdaBack = None
        self.n_env = n_env
        self.tf_ang_hf =  np.loadtxt(rf_path)
        self.tf = np.interp(angles, self.tf_ang_hf[:,1], self.tf_ang_hf[:,0]) / 1000
        self.hf = np.interp(angles, self.tf_ang_hf[:,1], self.tf_ang_hf[:,2])
        self.gaussAcc = self.gaussBeam(self.tf, args = {'z0': -0.0012, 'omega': 12 * 10 ** -6})
        pass

    def trace2ind(self, angleArray, timeArray, traceArray, args = {'v' : 1, 'l0': 0}):
        """
        Функция конвертации трейса светорассеяния частицы в индикатрису светорассеяния.
        Перевод осуществляется путем вычисления сплайна трейса светорассеяния I(θ), путем установления соответствия I(θ)<=>I(t) 
        и взятия интегралов I(θi) I(θi+1) для генарации сигнала в требуемых областях определения. 
        Для перевода обязательно ввести параметры инициализации цитометра (скорость потока и расстояние до триггера).  

        Parameters
        ----------
        angleArray : numpy.ndarray
            1D массив, набор углов, область определения индикатрис светорассеяния частиц 
        timeArray : numpy.ndarray
            1D массив, набор времен, область определения экспериментальных трейсов светорассеяния частиц 
        traceArray : numpy.ndarray
            2D массив, набор экспериментальных трейсов светорассеяния частиц
        args : dict
            Словарь параметров, с ключами 'v' (скорость потока) и 'l0' (расстояние до триггера), параметры инициализации цитометра   
        Returns
        -------
        indicatrix_array : numpy.ndarray
            Numpy 2D массив индикатрис светорассеяния частиц с областью определения angleArray
        """

        cords_array = timeArray * args['v'] + args['l0']
        anles_cord_array = np.interp(cords_array, self.tf, self.angles, left=0, right=0)
        indexes = np.array(range(len(timeArray)))[anles_cord_array != 0]

        indicatrix_array = []
        for i in range(len(traceArray)):
            indicatrix = []
            spline = InterpolatedUnivariateSpline(anles_cord_array[indexes], traceArray[i][indexes],  k=0)
            for j in range(len(angleArray)-1):
                indicatrix.append(spline.integral(angleArray[j], angleArray[j+1]))
            indicatrix.append(indicatrix[-1])
            indicatrix_array.append(indicatrix)
        indicatrix_array = np.array(indicatrix_array)
        gauss_hf_array = np.interp(angleArray, self.angles, self.gaussAcc) * np.interp(angleArray, self.angles, self.hf)
        return indicatrix_array / gauss_hf_array
        pass

    def ind2trace(self, ind, args):
        pass
    
    def gaussBeam(self, z, args):
        """
        Функция вычисления гауссового профиля лазерного луча. Требуется для точного вычисления обратной задачи.
        Конкретная функция вычисляет гауссовый профиль луча в одномерном случае при x=y=0.

        Parameters
        ----------
        z : numpy.ndarray
            1D массив, набор координат, область определения гауссового луча
        args : dict
            Словарь параметров, с ключами 'z0' (положение перетяжки), 'omega' (ширина перетяжки)

        Returns
        -------
        i : numpy.ndarray
            1D массив, вектор гауссового профиля луча нормированный на максимум
        """

        def om(_z, _omega):
            k = 2 * np.pi / (self.lambdaForw * 10 ** -9 / self.n_env)
            return _omega * (1 + ((2 * _z) / (k * _omega ** 2)) ** 2) ** 0.5
        i = 1 / (om(z - args['z0'], args['omega']) ** 2)
        return i / np.amax(i)

    def getSphereTrace(self, args):
        """
        Функция вычисления теоретического трейса светорассеяния для латексной микросферы с известными параметрами. 

        Parameters
        ----------
        args : dict
            Словарь параметров, с ключами:
                'd' (диаметр латексной микросферы)
                'n' (показатель преломления латексной микросферы)
                'v' (скорость потока)
                'l0' (расстояние до триггера)
        Returns
        -------
        time_array : numpy.ndarray
            1D массив, область определения трейса во времени
        hf_trace : numpy.ndarray
            1D массив, трейс светорассеяния латексной микросферы 
        """

        ind = self.getSphereInd(args)
        hf_trace = ind * self.hf * self.gaussAcc
        time_array = (-args['l0'] + self.tf) /  args['v']
        return time_array, hf_trace
        pass

    def getSphereInd(self, args):
        return miepython.ez_intensities(args['n'], args['d'], self.lambdaForw, self.mu, self.n_env, norm='bohren')[0]
        pass

    def initTraceSphereDB(self, time, DB_Division = 100000, args_range = 
    {
        'd' : [3690, 3700], 
        'n': [1.58, 1.6], 
        'v': [0.8, 1.2], 
        'l0': [-0.004, -0.003]
        }):

        x_DB = []
        y_DB = []
        
        for i in tqdm(range(DB_Division)):
            d = random.uniform(args_range['d'][0], args_range['d'][1])
            n = random.uniform(args_range['n'][0], args_range['n'][1])
            v = random.uniform(args_range['v'][0], args_range['v'][1])
            l0 = random.uniform(args_range['l0'][0], args_range['l0'][1])

            trace = self.getSphereTrace(args = {
                'd':d, 
                'n':n, 
                'v':v, 
                'l0':l0, 
                'z0': -0.0012, 
                'omega': 12 * 10 ** -6})

            x_DB.append(np.interp(time, trace[0], trace[1], left=0, right=0))
            y_DB.append([d, n, v, l0, -0.0012, 15 * 10 ** -9])

        return np.array(x_DB), np.array(y_DB)

    def initModel(self, x_DB, y_BD, method = 'brute_force', space = 'cosinesimil'):
        self.x_DB = x_DB
        self.y_DB = y_BD
        self.KNNModel = nmslib.init(method = method, space = space)
        self.KNNModel.addDataPointBatch(self.x_DB)
        self.KNNModel.createIndex()
        pass

    def KNNFit(self, x_Exp):
        res = self.KNNModel.knnQueryBatch(x_Exp, k=10)
        y_Exp = []
        indxs= []
        for i in range(len(res)):
            inds = res[i][0]
            y_Exp.append(self.y_DB[inds[0]])
            indxs.append(inds[0])
        y_Exp = np.array(y_Exp)
        indxs = np.array(indxs)
        return y_Exp, indxs
    
    def KNNSpheresFit(self, x_Exp):
        res = self.KNNModel.knnQueryBatch(x_Exp, k=10)
        y_Exp = []
        indxs= []
        for i in range(len(res)):
            inds = res[i][0]
            y_Exp.append(self.y_DB[inds[0]])
            indxs.append(inds[0])

        alphas = []
        for i in range(len(x_Exp)):
            alphas.append(np.dot(x_Exp[i], self.x_DB[indxs[i]]) / np.dot(self.x_DB[indxs[i]], self.x_DB[indxs[i]]))

        y_Exp = np.array(y_Exp)
        indxs = np.array(indxs)

        y_Exp_Dict = {
            'd':y_Exp[:,0],
            'n':y_Exp[:,1],
            'v':y_Exp[:,2],
            'l0':y_Exp[:,3],
            'alpha':alphas}
        return y_Exp_Dict, indxs
