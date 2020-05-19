from netCDF4 import Dataset
import numpy as np
import cv2
import glob, os
import sys
import warnings

warnings.filterwarnings("ignore")


class GaussianNB:
    """
    Класс байесовского классификатора
    """

    def __init__(self, X, y):
        """
        функция создания и обучения модели
        :param X: входные данные
        :param y: выходные данные
        """
        self.classes = np.unique(y)
        shape = (2, X.shape[1])
        self.theta_ = np.zeros(shape)
        self.sigma_ = np.zeros(shape)
        self.class_count_ = np.zeros(2, dtype=np.float64)
        for y_i in self.classes:
            i = self.classes.searchsorted(y_i)
            X_i = X[y == y_i, :]
            self.theta_[i, :] = np.mean(X_i)
            self.sigma_[i, :] = np.var(X_i)
            self.class_count_[i] += X_i.shape[0]
        self.sigma_[:, :] += 1e-9 * np.var(X).max()
        self.class_prior_ = self.class_count_ / self.class_count_.sum()

    def predict(self, X):
        """
        Функция предсказания результата
        :param X: входные данные
        :return: Предсказание
        """
        joint_log_likelihood = [np.log(self.class_prior_[i]) -
                                0.5 * (np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
                                       + np.sum(((X - self.theta_[i, :]) ** 2) / (self.sigma_[i, :]), 1))
                                for i in range(2)]
        return self.classes[np.argmax(np.array(joint_log_likelihood).T, axis=1)]


class Channel:
    def __init__(self, name, longname, picture, flags="", wavelength=-1):
        """
        Класс хранения одного кадра спутниковой сцены
        :param name: Короткое название канала
        :param longname: Полное название канала
        :param picture: Изображение с канала
        :param flags: Флаги, если канал размечен
        :param wavelength: Длина волны, если имеется
        """
        self.name = name
        self.longname = longname
        self.picture = picture
        self.flags = flags
        self.wavelength = wavelength
        self.updateMeta()

    def updateMeta(self):
        """
        Обновляет метаинформацию о канале
        min - минимаельное значение пикселя канала
        max - максимальное значение пикселя канала
        shape - форма изображенния канала
        :return:
        """
        self.min = np.min(self.picture)
        self.max = np.max(self.picture)
        self.shape = self.picture.shape

    def normalise(self):
        """
        Сдвигает спектр изображения канала до интервала [0; 255]
        :return:
        """
        self.picture -= self.min
        self.picture = self.picture * 255 / np.max(self.picture)
        self.updateMeta()

    def save(self, folder, printMeta=True):
        """
        Сохраняет изображение с канала как картинку
        :param folder: Путь до папки сохранения
        :param printMeta: Нужно ли печатать информацию об успешности сохранения
        :return:
        """
        try:
            cv2.imwrite(folder + "/" + self.longname + ".jpg", self.picture)
            if printMeta: print(self.longname + " saved in " + folder)
        except:
            if printMeta: print("Can't save " + self.name, file=sys.stderr)

    def printMeta(self):
        """
        печатает метаинформацию о канале
        :return:
        """
        print(self.name, self.longname, '\n',
              'min:', self.min,
              'max:', self.max,
              'size: ', self.shape,
              'wavelength', self.wavelength)


class Scene:
    def __init__(self, filename, onlyPic=False, onlyFull=False, printMeta=False):
        """
        Класс спутниковой сцены, которая хранит информацию об изображениях в отдельных каналах
        :param filename: название файла, хранящего спутниковую сцену
        :param onlyPic: True, если нужно обрабатывать только каналы с картинкой
        :param onlyFull: True, если нужно только каналы с полным изображением
        """
        self.channels = {}
        nc = Dataset(filename, "r")
        for channelName in nc.variables:
            if printMeta: print(nc.variables[channelName].long_name)
            flags = ''
            wavelength = -1
            try:
                flags = nc.variables[channelName].flag_meanings
            except Exception as e:
                if printMeta: print('Empty flags')
            try:
                wavelength = nc.variables[channelName].wavelength
            except Exception as e:
                if printMeta: print('Empty wavelength')
            channelPicture = np.array(nc.variables[channelName])
            channel = Channel(name=nc.variables[channelName].name,
                              longname=nc.variables[channelName].long_name,
                              picture=channelPicture, wavelength=wavelength,
                              flags=flags)
            if (len(channel.shape) > 1 or not onlyPic) and \
                    (channel.shape[0] - channel.shape[1]) * 10 < channel.shape[0] or not onlyFull:
                self.channels[channelName] = channel


class FullScene:
    def __init__(self, folder):
        self.scenes = {}
        self.folder = folder
        for file in glob.glob(folder + "/*.nc"):
            scene = Scene(file, onlyPic=True, onlyFull=True)
            self.scenes[file] = scene
        self.general_mask = self.scenes[folder + '\\flags.nc'].channels['OLC_flags'].picture
        self.cloud_mask = self.scenes[folder + '\\flags.nc'].channels['CLOUD_flags'].picture
        self.cloud_land_mask_sensor = self.scenes[folder + '\\Syn_Oa10_reflectance.nc'].channels['SDR_Oa10_err'].picture
        self.landmask = np.zeros(self.general_mask.shape)
        self.landmask[self.general_mask // 4096 % 2 == 1] = 1  # OLC_land
        self.landmask[self.general_mask // 1024 % 2 == 1] = 0  # OLC_fresh_inland_water

    def saveOriginal(self, filename):
        resultpic = np.full((self.general_mask.shape + (3,)), [255, 187, 153])  # Sea
        resultpic[self.general_mask // 4096 % 2 == 1] = [0, 200, 0]  # OLC_land
        resultpic[self.general_mask // 1024 % 2 == 1] = [100, 0, 0]  # OLC_fresh_inland_water
        resultpic[self.cloud_mask % 2 == 1] = [250, 250, 250]
        self.orig_pic = resultpic
        cv2.imwrite(filename, resultpic)

    def saveIFmteod(self, filename):
        landpik = np.full((self.general_mask.shape + (3,)), [255, 187, 153])
        landpik[self.general_mask // 4096 % 2 == 1] = [0, 200, 0]  # OLC_land
        landpik[self.general_mask // 1024 % 2 == 1] = [100, 0, 0]  # OLC_fresh_inland_water
        cloud_land_mask_sensor = self.cloud_land_mask_sensor.copy()
        cloud_mask_copy = self.cloud_mask.copy()
        cloud_mask_copy[self.landmask == 1] = 0
        landpik[cloud_mask_copy % 2 == 1] = [250, 250, 250]
        cloud_land_mask_sensor[self.landmask == 0] = 1000
        landpik[cloud_land_mask_sensor < 100] = [250, 250, 250]
        self.if_pic = landpik
        cv2.imwrite(filename, landpik)

    def saveAImetod(self, filename, model, isModelEmpty=False):
        cloud_mask = self.cloud_mask.copy()
        cloud_mask[cloud_mask % 2 == 1] = 1
        cloud_mask[cloud_mask % 2 == 0] = 0
        if isModelEmpty:
            cloud_arr = np.concatenate(self.cloud_land_mask_sensor).reshape(-1, 1)
            cloud_test_mask = np.concatenate(cloud_mask)
            model = GaussianNB(cloud_arr, cloud_test_mask)
        ai_mask = model.predict(np.concatenate(self.cloud_land_mask_sensor).reshape(-1, 1)).reshape(-1, self.general_mask.shape[1])
        landpik = np.full((self.general_mask.shape + (3,)), [255, 187, 153])
        landpik[self.general_mask // 4096 % 2 == 1] = [0, 200, 0]  # OLC_land
        landpik[self.general_mask // 1024 % 2 == 1] = [100, 0, 0]  # OLC_fresh_inland_water
        cloud_mask_copy = cloud_mask.copy()
        cloud_mask_copy[self.landmask == 1] = 0
        landpik[cloud_mask_copy % 2 == 1] = [250, 250, 250]
        ai_pic = landpik.copy()
        ai_mask[self.landmask == 0] = 0
        ai_pic[ai_mask == 1] = [250, 250, 250]
        self.ai_pic=ai_pic
        cv2.imwrite(filename, ai_pic)
        return model


krym = FullScene('last_test_2')
krym.saveOriginal('krum_orig.jpg')
krym.saveIFmteod('krum_IF.jpg')
model = krym.saveAImetod(filename="ktym_AI.jpg", model=None, isModelEmpty=True)


print(np.mean(krym.orig_pic != krym.if_pic))
print(np.mean(krym.orig_pic != krym.ai_pic))


italy = FullScene('last_test_1')
italy.saveOriginal('italy_orig.jpg')
italy.saveIFmteod('italy_IF.jpg')
italy.saveAImetod(filename="italy_AI.jpg", model=model, isModelEmpty=False)

print(np.mean(italy.orig_pic != italy.if_pic))
print(np.mean(italy.orig_pic != italy.ai_pic))
