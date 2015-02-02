# Jordan Makansi
# CEE 263N Project

# Imports and Global variables 
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
from osgeo import gdal
from gdal import gdalconst
from gdalconst import *
import pandas as pd 

#Data is 1 squared km, so each cell is 1 squared meter 

gdal.AllRegister()
dataset = gdal.Open('jordan_snow.tif') # This picture is from Zeshi 
snow_array = dataset.ReadAsArray()
geo = dataset.GetGeoTransform()
topleftX = geo[0]
topleftY = geo[3]
cellSizeX = geo[1]
cellSizeY = geo[5]

