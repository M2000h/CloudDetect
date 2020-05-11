from netCDF4 import Dataset
import numpy as np
import cv2
import glob, os
from zipfile import ZipFile
import sys

import warnings

warnings.filterwarnings("ignore")


class Channel:
    def __init__(self, name, picture):
        self.name = name
        self.picture = picture
        self.updateMeta()

    def updateMeta(self):
        self.min = np.min(self.picture)
        self.max = np.max(self.picture)
        self.shape = self.picture.shape

    def normalise(self):
        self.picture -= self.min
        self.picture = self.picture * 256 / np.max(self.picture)
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
              'size: ', self.shape,
              'std', np.std(self.picture))


class Scene:
    def __init__(self, filename, onlyPic=False, onlyFull=False):
        self.channels = {}

        nc = Dataset(filename, "r")
        for channelName in nc.variables:
            channelPicture = np.array(nc.variables[channelName])
            channel = Channel(name=file + "_" + channelName,
                              picture=channelPicture)
            if (len(channel.shape) > 1 or not onlyPic) and \
                    (channel.shape[0] - channel.shape[1]) * 10 < channel.shape[0] or not onlyFull:
                self.channels[channelName] = channel


# with ZipFile('kek.zip', 'r') as zip:
#     # printing all the contents of the zip file
#     zip.printdir()
#     a = zip.open('kek/flags.nc')# .read('kek/flags.nc')
#     b = open('waaa.nc', 'wb')
#     b.write(a.read())
#     b.close()
#     nc = Dataset('waaa.nc', 'r')
#     for kek in  nc.variables:
#         kk = np.array(nc.variables[kek])
#         ch = Channel(name=kek, picture=kk)
#         ch.save("pics")
#     input('>>>')
#     topo = nc.variables['CLOUD_flags']
#     try:
#             kk = np.array(topo)
#             if np.amax(kk) < 20:
#                 pass
#                 kk = kk * 50
#                 print("use max")
#             print(kk.shape)
#             print(np.amax(kk), np.mean(kk), np.amin(kk))
#             cv2.imshow("file" + " " + str("topo1"), kk)
#             cv2.imwrite("output.jpg", kk)
#             cv2.waitKey()
#     except Exception as e:
#             pass
#     print(nc.variables.keys())
#     topo1 = "OLC_flags"
#     topo = nc.variables[topo1]
#     kk = np.array(topo)
#     land_pic = np.zeros((4091, 4865, 3))
#     # land_pic = np.where(kk%4096==0, [0, 200, 0], land_pic)
#     land_pic[kk >= 4096] = [0, 200, 0]
#     kk[kk >= 4096] -= 4096
#     land_pic[kk >= 2048] = [200, 0, 0]
#     kk[kk >= 2048] -= 2048
#     land_pic[kk >= 1024] = [200, 0, 0]
#     kk[kk >= 1024] -= 1024
#     land_pic[kk >= 512] = [100, 0, 0]
#     kk[kk >= 512] -= 512
#     land_pic[kk >= 256] = [200, 0, 200]
#     kk[kk >= 256] -= 256
#     land_pic[kk >= 128] = [0, 0, 200]
#     # Blue Green Red
#     cv2.imshow("file" + " " + str(topo1), land_pic)
#     cv2.namedWindow("file" + " " + str(topo1), cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("file" + " " + str(topo1), 800, 600)
#     cv2.imwrite("output1.jpg", land_pic)
#     cv2.waitKey()

# input('>>>')
os.chdir("1")
for file in glob.glob("*.nc"):
    scene = Scene(file, onlyPic=True, onlyFull=True)
    for channel in scene.channels.values():
        channel.normalise()
        channel.save("../pics", printMeta=False)
        channel.printMeta()

"""nc = Dataset(file, "r")
    for topo1 in nc.variables:
        print()
        print(topo1)
        topo = nc.variables[topo1]
        # topo = nc.variables['CLOUD_flags']
        try:
            kk = np.array(topo)
            # kk %= 256
            # if np.max(kk) < 20:
            #     pass
            #     kk = kk * 50
            #     print("use max")

            print(kk.shape)
            # if np.amax(kk) > 255:
            #     kk = kk // (np.amax(kk) / 200)
            # print(np.max(kk), np.mean(kk), np.min(kk))
            if topo1 == "OLC_flags":
                land_pic = np.zeros((kk.shape + (3,)))
                # land_pic = np.where(kk%4096==0, [0, 200, 0], land_pic)
                land_pic[kk >= 4096] = [0, 200, 0]
                kk[kk >= 4096] -= 4096
                land_pic[kk >= 2048] = [200, 0, 0]
                kk[kk >= 2048] -= 2048
                land_pic[kk >= 1024] = [200, 0, 0]
                kk[kk >= 1024] -= 1024
                land_pic[kk >= 512] = [100, 0, 0]
                kk[kk >= 512] -= 512
                land_pic[kk >= 256] = [200, 0, 200]
                kk[kk >= 256] -= 256
                land_pic[kk >= 128] = [0, 0, 200]
                # Blue Green Red
                cv2.imshow("file" + " " + str(topo1), land_pic)
                cv2.namedWindow("file" + " " + str(topo1), cv2.WINDOW_NORMAL)
                cv2.resizeWindow("file" + " " + str(topo1), 800, 600)
                cv2.waitKey()
                cv2.imwrite("output1.jpg", land_pic)
                channel = Channel(name=file + "_" + str(topo1), picture=kk)
                channel.save(folder="../pics")
            else:
                channel = Channel(name=file + "_" + str(topo1), picture=kk)
                channel.save(folder="../pics")
                # cv2.imshow("file" + " " + str(topo1), kk)
                # cv2.waitKey()
        except Exception as e:
            print("ERRRRRRRRRRRRRRRRRROOOOOOOOOOOOOOOOOOOOOOORRRRRRRRRRRRRRRRRRRRRRR")
            print(e)"""
