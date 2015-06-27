# Jordan Makansi
# importing the kml file and the tif file 

from fastkml import kml
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
from osgeo import gdal
from gdal import gdalconst
from gdalconst import *
import sys
import time 

#### ----------- Importing using fastkml --------------

doc = file('nodes\\doc.kml').read()
k = kml.KML()
k.from_string(doc)
len(k.features())
