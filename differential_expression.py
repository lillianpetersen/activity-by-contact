import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
from scipy import stats
from collections import Counter

wd = '/pbld/mcg/lillianpetersen/ABC/'

print Type
peak_file = wd+'peak_allignment/with_control/'+Type+'/'
if Type=='B_ALL':
	groups = np.array([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]) # 0=control 1=cancer

peakFile = np.array(pd.read_csv(peak_file+Type+'_merged_counts.rpkm.qn.txt', sep = '\t', header = None))
peakFile = np.swapaxes(peakFile,0,1)
allPeaks1 = np.array(peakFile[1:], dtype = float)
allPeaksAvg1 = np.mean(allPeaks, axis = 0)

# Keep only large enough peaks
keep = allPeaks[2:]>(np.mean(allPeaks[2:])-1)
keep = np.sum(keep,axis=0) >= ((len(groups)-1)/2)
keep = 1-keep

controlPeaksAll = np.array(allPeaks[groups==0], dtype = float)
cancerPeaksAll = np.array(allPeaks[groups==1], dtype = float)
peakNames1 = peakFile[0]

controlPeaks1 = np.mean(controlPeaksAll,axis=0)
cancerPeaks1 = np.mean(cancerPeaksAll,axis=0)

cancerPeaks = np.ma.compressed(np.ma.masked_array(cancerPeaks1,keep))
allPeaksAvg = np.ma.compressed(np.ma.masked_array(allPeaksAvg1,keep))
peakNames = np.ma.compressed(np.ma.masked_array(peakNames1,keep))
controlPeaks = np.ma.compressed(np.ma.masked_array(controlPeaks1,keep))

cancerPeaks = np.ma.compressed(np.ma.masked_array(cancerPeaks,controlPeaks==0))
allPeaksAvg = np.ma.compressed(np.ma.masked_array(allPeaksAvg,controlPeaks==0))
peakNames = np.ma.compressed(np.ma.masked_array(peakNames,controlPeaks==0))
controlPeaks = np.ma.compressed(np.ma.masked_array(controlPeaks,controlPeaks==0))

if len(controlPeaks)==len(cancerPeaks)==len(peakNames) == False:
	print 'error: arrays are different lengths!'
	exit()

foldChange = np.log2(cancerPeaks/controlPeaks)
rpkm = np.log2(allPeaksAvg)

plt.clf()
plt.scatter(rpkm, foldChange, s=1, c='k')
plt.plot([1,8],[-2,-2],'b-', linewidth = 2)
plt.plot([1,8],[2,2],'b-', linewidth = 2)
plt.plot([1,8],[0,0],'k-', linewidth = 1)
plt.title('B ALL: Differential Expression of Peaks', fontsize = 20)
plt.xlabel('Average log2(rpkm)')
plt.ylabel('Average log2(Fold Change)')
plt.ylim([-5,10])
plt.xlim([1,8])
plt.grid(True)
plt.show()
exit()




