from netCDF4 import Dataset
import numpy as np
import cv2
import glob, os
from zipfile import ZipFile
import sys

import warnings

warnings.filterwarnings("ignore")


class Channel:
    def __init__(self, name, longname, picture, flags="", wavelength=-1):
        self.name = name
        self.longname = longname
        self.picture = picture
        self.flags = flags
        self.wavelength = wavelength
        self.updateMeta()

    def updateMeta(self):
        self.min = np.min(self.picture)
        self.max = np.max(self.picture)
        self.shape = self.picture.shape

    def normalise(self):
        self.picture -= self.min
        self.picture = self.picture * 255 / np.max(self.picture)
        self.updateMeta()

    def save(self, folder, printMeta=True):
        try:
            cv2.imwrite(folder + "/" + self.name + ".jpg", self.picture)
            if printMeta: print(self.name + " saved in " + folder)
        except:
            if printMeta: print("Can't save " + self.name, file=sys.stderr)

    def printMeta(self):
        print(self.name, self.longname, '\n',
              'min:', self.min,
              'max:', self.max,
              'size: ', self.shape, )


class Scene:
    def __init__(self, filename, onlyPic=False, onlyFull=False):
        self.channels = {}
        nc = Dataset(filename, "r")
        for channelName in nc.variables:
            print(nc.variables[channelName].long_name)
            flags = ''
            wavelength = -1
            try:
                flags = nc.variables[channelName].flag_meanings
            except Exception as e:
                print('Empty flags')
            try:
                wavelength = nc.variables[channelName].wavelength
            except Exception as e:
                print('Empty wavelength')
            channelPicture = np.array(nc.variables[channelName])
            channel = Channel(name=nc.variables[channelName].name,
                              longname=nc.variables[channelName].long_name,
                              picture=channelPicture, wavelength=wavelength,
                              flags=flags)
            if (len(channel.shape) > 1 or not onlyPic) and \
                    (channel.shape[0] - channel.shape[1]) * 10 < channel.shape[0] or not onlyFull:
                self.channels[channelName] = channel


scenes = {}
os.chdir("11")
for file in glob.glob("flags.nc"):
    scene = Scene(file, onlyPic=True, onlyFull=True)
    # for channel in scene.channels.values():
    #     channel.normalise()
    #     channel.save("../pics", printMeta=False)
    #     channel.printMeta()
    scenes[file] = scene
for file in glob.glob("Syn_Oa10_reflectance.nc"):
    scene = Scene(file, onlyPic=True, onlyFull=True)
    scenes[file] = scene

# input('>>>')

for flag in scenes['flags.nc'].channels:
    print(flag, scenes['flags.nc'].channels[flag].name, scenes['flags.nc'].channels[flag].flags)

general_mask = scenes['flags.nc'].channels['OLC_flags'].picture
cloud_mask = scenes['flags.nc'].channels['CLOUD_flags'].picture
cloud_land_mask = scenes['Syn_Oa10_reflectance.nc'].channels['SDR_Oa10_err'].picture

resultpic = np.full((general_mask.shape + (3,)), [255, 187, 153])
resultpic[general_mask // 4096 % 2 == 1] = [0, 200, 0]  # OLC_land
resultpic[general_mask // 1024 % 2 == 1] = [100, 0, 0]  # OLC_fresh_inland_water
resultpic[cloud_mask % 2 == 1] = [250, 250, 250]
cv2.imwrite("../result.jpg", resultpic)

landpik = np.full((general_mask.shape + (3,)), [255, 187, 153])
landmask = np.zeros(general_mask.shape)

landpik[general_mask // 4096 % 2 == 1] = [0, 200, 0]  # OLC_land
landpik[general_mask // 1024 % 2 == 1] = [100, 0, 0]  # OLC_fresh_inland_water
landmask[general_mask // 4096 % 2 == 1] = 1
landmask[general_mask // 1024 % 2 == 1] = 0
cloud_mask[landmask == 1] = 0
landpik[cloud_mask % 2 == 1] = [250, 250, 250]
cloud_land_mask[landmask == 0] = 1000
landpik[cloud_land_mask < 100] = [250, 250, 250]
cv2.imwrite("../land.jpg", landpik)

print(np.mean(resultpic != landpik))