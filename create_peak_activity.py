import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
from scipy import stats
from collections import Counter
import statsmodels.stats.multitest as multi
from math import e
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sklearn.linear_model

wd = '/pbld/mcg/lillianpetersen/ABC/'
wdvars = '/pbld/mcg/lillianpetersen/ABC/saved_variables/'
wdfigs = '/pbld/mcg/lillianpetersen/ABC/figures/'
MakePlots = False

###################################################################
# Load ATAC
###################################################################
print('Load ATAC')
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
    if not 'atac'+ichr in globals():
        vars()['atac'+ichr] = np.array(np.load(wdvars+'validation_K562/ATAC/atac'+ichr+'.npy'),dtype=float)
        vars()['chrATAC'+ichr] = np.load(wdvars+'validation_K562/ATAC/chrATAC'+ichr+'.npy')
        vars()['peakName'+ichr] = np.load(wdvars+'validation_K562/ATAC/peakName'+ichr+'.npy')
        vars()['positionATAC'+ichr] = np.load(wdvars+'validation_K562/ATAC/positionATAC'+ichr+'.npy')
        vars()['nPeaks'+ichr] = np.load(wdvars+'validation_K562/ATAC/nPeaks'+ichr+'.npy')

###################################################################
# Load Chip
###################################################################
print('Load Chip')
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
    if not 'chip'+ichr in globals():
        vars()['chip'+ichr] = np.array(np.load(wdvars+'validation_K562/Chip/chip'+ichr+'.npy'),dtype=float)
        vars()['chrChip'+ichr] = np.load(wdvars+'validation_K562/Chip/chrChip'+ichr+'.npy')
        vars()['peakNameChip'+ichr] = np.load(wdvars+'validation_K562/Chip/peakName'+ichr+'.npy')
        vars()['positionChip'+ichr] = np.load(wdvars+'validation_K562/Chip/positionChip'+ichr+'.npy')
        vars()['nPeaksChip'+ichr] = np.load(wdvars+'validation_K562/Chip/nPeaks'+ichr+'.npy')

###################################################################
# Create Activity
###################################################################
print('Create Activity')
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	chrLen = np.amax( [np.amax(vars()['positionATAC'+ichr][1]), np.amax(vars()['positionChip'+ichr][1])] )
	chipPos = -9999*np.ones(shape=chrLen,dtype=int)
	for ichip in range( vars()['nPeaksChip'+ichr] ):
		chipPos[ vars()['positionChip'+ichr][0,ichip]:vars()['positionChip'+ichr][1,ichip] ] = ichip

	vars()['activity'+ichr] = np.zeros(shape=(vars()['atac'+ichr].shape))
	goodChip = 0
	noChip = 0
	for ipeak in range( vars()['nPeaks'+ichr] ):
		ichip = np.amax( chipPos[vars()['positionATAC'+ichr][0,ipeak]:vars()['positionATAC'+ichr][1,ipeak]+1] )

		if ichip>-1 and vars()['chip'+ichr][ichip]>0:
			vars()['activity'+ichr][ipeak] = np.sqrt( vars()['atac'+ichr][ipeak] * vars()['chip'+ichr][ichip] )
			goodChip+=1
		else:
			vars()['activity'+ichr][ipeak] = np.sqrt(vars()['atac'+ichr][ipeak])
			noChip+=1
		
###################################################################
# Create Activity
###################################################################
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
    np.save(wdvars+'validation_K562/ATAC/activity'+ichr+'.npy', vars()['activity'+ichr])

