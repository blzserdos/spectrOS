import numpy as np
import pandas as pd
from uv_nir_gos	import SNV, load_data, train_test_split, wl_select, plsregress, PLSR_test, build_model, train_ann, ANN_test

raw, NIR, UV, full_r, full = load_data(snv_corr=True)

test_i, Train, Test = train_test_split(full, test='E')

print('Start PLSR')

print('Train UV-PLSR')
plsregress(Train, Test, 9, spec='UV')
resDF_UV = PLSR_test(Train, Test, 10, spec='UV')

print('Train UVr-PLSR')
plsregress(Train, Test, 9, spec='UVr')
resDF_UVr = PLSR_test(Train, Test, 10, spec='UVr')

print('Train NIR-PLSR')
plsregress(Train, Test, 13, spec='NIR')
resDF_NIR = PLSR_test(Train, Test, 14, spec='NIR')

print('Train ALLr-PLSR')
plsregress(Train, Test, 15, spec='ALLr')
resDF_ALLr = PLSR_test(Train, Test, 16, spec='ALLr')

print('Train ALL-PLSR')
plsregress(Train, Test, 15, spec='ALL')
resDF_ALL = PLSR_test(Train, Test, 16, spec='ALL')

"""
print('Train UVr-ANN')
layerszs = [32, 16]
train_ann(Train, Test, [layerszs[0],layerszs[1]], opti='rmsprop', spec='UVr', modeltype='1dconv')
"""

print('Start ANN')

print('Train UVr-ANN')
layerszs = [32, 16]
train_ann(Train, Test, [layerszs[0],layerszs[1]], opti='rmsprop', spec='UVr', modeltype='ff')
print('test UVr-ANN')
resDF_UVr = ANN_test(Train, Test, [layerszs[0],layerszs[1]], spec='UV',modeltype='ff')

print('Train UV-ANN')
train_ann(Train, Test, [layerszs[0],layerszs[1]], opti='rmsprop', spec='UV', modeltype='ff')
print('test UV-ANN')
resDF_UV = ANN_test(Train, Test, [layerszs[0],layerszs[1]], spec='UV',modeltype='ff')

#layerszs = [64, 32]
print('Train NIR-ANN')
train_ann(Train, Test, [layerszs[0],layerszs[1]], opti='rmsprop', spec='NIR', modeltype='ff')
print('test NIR-ANN')
resDF_NIR = ANN_test(Train, Test, [layerszs[0],layerszs[1]], spec='NIR',modeltype='ff')

print('Train ALL-ANN')
train_ann(Train, Test, [layerszs[0],layerszs[1]], opti='rmsprop', spec='ALL', modeltype='ff')
print('test ALL-ANN')
resDF_ALL = ANN_test(Train, Test, [layerszs[0],layerszs[1]], spec='ALL',modeltype='ff')

print('Train ALLr-ANN')
train_ann(Train, Test, [layerszs[0],layerszs[1]], opti='rmsprop', spec='ALLr', modeltype='ff')
print('test ALLr-ANN')
resDF_ALLr = ANN_test(Train, Test, [layerszs[0],layerszs[1]], spec='ALLr',modeltype='ff')
