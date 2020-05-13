from netCDF4 import Dataset
import numpy as np
import cv2
import glob, os
from zipfile import ZipFile
import sys

import warnings

warnings.filterwarnings("ignore")


class Channel:
    def __init__(self, name, picture, flags="", wavelength=-1):
        self.name = name
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
        print(self.name, '\n',
              'min:', self.min,
              'max:', self.max,
              'size: ', self.shape, )


class Scene:
    def __init__(self, filename, onlyPic=False, onlyFull=False):
        self.channels = {}
        nc = Dataset(filename, "r")
        print()
        print()
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
            print()
            channelPicture = np.array(nc.variables[channelName])
            channel = Channel(name=nc.variables[channelName].long_name,
                              picture=channelPicture, wavelength=wavelength,
                              flags=flags)
            if (len(channel.shape) > 1 or not onlyPic) and \
                    (channel.shape[0] - channel.shape[1]) * 10 < channel.shape[0] or not onlyFull:
                self.channels[channelName] = channel


scenes = {}
os.chdir("3")
for file in glob.glob("flags.nc"):
    scene = Scene(file, onlyPic=True, onlyFull=True)
    for channel in scene.channels.values():
        # channel.normalise()
        channel.save("../pics", printMeta=False)
        # channel.printMeta()
    scenes[file] = scene

# for flag in scenes['flags.nc'].channels.values():
#     print(flag.flags)
print(scenes['flags.nc'].channels['OLC_flags'].flags)
cloudmask = scenes['flags.nc'].channels['OLC_flags'].picture
resultpic = np.full((cloudmask.shape + (3,)),  [255, 187, 153])
print(np.max(cloudmask))
resultpic[cloudmask >= 4096] = [0, 200, 0]
cloudmask %= 4096
print(np.max(cloudmask))
resultpic[cloudmask >= 2048] = [200, 0, 0]
cloudmask %= 2048
print(np.max(cloudmask))
resultpic[cloudmask >= 1024] = [200, 0, 0]
cloudmask %= 1024
print(np.max(cloudmask))
resultpic[cloudmask >= 512] = [100, 0, 0]
cloudmask %= 512
print(np.max(cloudmask))
resultpic[cloudmask >= 256] = [250, 250, 250]
cloudmask %= 256
print(np.max(cloudmask))
resultpic[cloudmask >= 128] = [0, 0, 200]
cv2.imwrite("../output1.jpg", resultpic)
