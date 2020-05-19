from netCDF4 import Dataset
import numpy as np
import cv2
import glob, os
import sys
import warnings

warnings.filterwarnings("ignore")


class GaussianNB():
    def __init__(self):
        pass

    def fit(self, X, y):
        return self._partial_fit(X, y, np.unique(y))

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def _partial_fit(self, X, y, classes=None):
        self.epsilon_ = 1e-9 * np.var(X, axis=0).max()
        self.classes_ = classes
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.class_prior_ = np.zeros(n_classes, dtype=np.float64)
        unique_y = np.unique(y)
        for y_i in unique_y:
            i = classes.searchsorted(y_i)
            X_i = X[y == y_i, :]
            N_i = X_i.shape[0]
            self.theta_[i, :] = np.mean(X_i, axis=0)
            self.sigma_[i, :] = np.var(X_i, axis=0)
            self.class_count_[i] += N_i
        self.sigma_[:, :] += self.epsilon_
        self.class_prior_ = self.class_count_ / self.class_count_.sum()
        return self

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) /
                                 (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)
        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood


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


scenes = {}
os.chdir("last_test_2")
for file in glob.glob("*.nc"):
    scene = Scene(file, onlyPic=True, onlyFull=True)
    # for channel in scene.channels.values():
    #     channel.normalise()
    #     channel.save("../pics", printMeta=False)
    #     channel.printMeta()
    scenes[file] = scene
for file in glob.glob("Syn_Oa10_reflectance.nc"):
    scene = Scene(file, onlyPic=True, onlyFull=True)
    scenes[file] = scene

general_mask = scenes['flags.nc'].channels['OLC_flags'].picture
cloud_mask = scenes['flags.nc'].channels['CLOUD_flags'].picture
cloud_land_mask_sensor = scenes['Syn_Oa10_reflectance.nc'].channels['SDR_Oa10_err'].picture

resultpic = np.full((general_mask.shape + (3,)), [255, 187, 153])
resultpic[general_mask // 4096 % 2 == 1] = [0, 200, 0]  # OLC_land
resultpic[general_mask // 1024 % 2 == 1] = [100, 0, 0]  # OLC_fresh_inland_water
resultpic[cloud_mask % 2 == 1] = [250, 250, 250]
cv2.imwrite("../result.jpg", resultpic)

landpik = np.full((general_mask.shape + (3,)), [255, 187, 153])
landmask = np.zeros(general_mask.shape)

landpik[general_mask // 4096 % 2 == 1] = [0, 200, 0]  # OLC_land
landpik[general_mask // 1024 % 2 == 1] = [100, 0, 0]  # OLC_fresh_inland_water
landmask[general_mask // 4096 % 2 == 1] = 1  # OLC_land
landmask[general_mask // 1024 % 2 == 1] = 0  # OLC_fresh_inland_water
cloud_mask_copy = cloud_mask.copy()
cloud_mask_copy[landmask == 1] = 0
landpik[cloud_mask_copy % 2 == 1] = [250, 250, 250]

cloud_mask[cloud_mask % 2 == 1] = 1
cloud_mask[cloud_mask % 2 == 0] = 0

ai_pic = landpik.copy()
model = GaussianNB()
landmask_arr = np.concatenate(landmask)
cloud_arr = np.concatenate(cloud_land_mask_sensor).reshape(-1, 1)
cloud_test_mask = np.concatenate(cloud_mask)

print('start fit')
model.fit(cloud_arr, cloud_test_mask)

i = 0

print('start predict')
ai_mask = model.predict(np.concatenate(cloud_land_mask_sensor).reshape(-1, 1)).reshape(-1, general_mask.shape[1])

ai_mask[landmask == 0] = 0
ai_pic[ai_mask == 1] = [250, 250, 250]

cloud_land_mask_sensor[landmask == 0] = 1000
landpik[cloud_land_mask_sensor < 100] = [250, 250, 250]
cv2.imwrite("../land.jpg", landpik)

cv2.imwrite("../ai_pic.jpg", ai_pic)

print(np.mean(resultpic != landpik))
print(np.mean(resultpic != ai_pic))
