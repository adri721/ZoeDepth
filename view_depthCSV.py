import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import time
import cv2
import pgm_reader


rows=[]
filename = "C:/Users/ARajaraman/OneDrive - SharkNinja/Documents/pythontests/ZoeDepth/Intel_data/2024_4_17_17_2_3/depthmap/0.csv"
# reader = pgm_reader.Reader()
# depthmap = reader.read_pgm(filename)

depthmap = np.genfromtxt(filename, delimiter=',')
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthmap, alpha=0.35), cv2.COLORMAP_INFERNO)
cv2.imshow('depth', depth_colormap)
cv2.waitKey(0)