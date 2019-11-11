import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
from scipy import stats
from collections import Counter
import statsmodels.stats.multitest as multi
from collections import defaultdict
import re
import gzip

wd = '/pbld/mcg/lillianpetersen/ABC/'
wdvars = '/pbld/mcg/lillianpetersen/ABC/saved_variables/'
wdfigs = '/pbld/mcg/lillianpetersen/ABC/figures/'
wddata = '/pbld/mcg/lillianpetersen/ABC/data/'
MakePlots = False

rnaGenes = np.load(wdvars+'geneNameAll.npy')


GTF_HEADER  = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame']
R_SEMICOLON = re.compile(r'\s*;\s*')
R_COMMA     = re.compile(r'\s*,\s*')
R_KEYVALUE  = re.compile(r'(\s+|\s*=\s*)')


def dataframe(filename):
    """Open an optionally gzipped GTF file and return a pandas.DataFrame.
    """
    # Each column is a list stored as a value in this dict.
    result = defaultdict(list)

    for i, line in enumerate(lines(filename)):
        for key in line.keys():
            # This key has not been seen yet, so set it to None for all
            # previous lines.
            if key not in result:
                result[key] = [None] * i

        # Ensure this row has some value for each column.
        for key in result.keys():
            result[key].append(line.get(key, None))

    return pd.DataFrame(result)


def lines(filename):
    """Open an optionally gzipped GTF file and generate a dict for each line.
    """
    fn_open = gzip.open if filename.endswith('.gz') else open

    with fn_open(filename) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            else:
                yield parse(line)


def parse(line):
    """Parse a single GTF line and return a dict.
    """
    result = {}

    fields = line.rstrip().split('\t')

    for i, col in enumerate(GTF_HEADER):
        result[col] = _get_value(fields[i])

    # INFO field consists of "key1=value;key2=value;...".
    infos = [x for x in re.split(R_SEMICOLON, fields[8]) if x.strip()]

    for i, info in enumerate(infos, 1):
        # It should be key="value".
        try:
            key, _, value = re.split(R_KEYVALUE, info, 1)
        # But sometimes it is just "value".
        except ValueError:
            key = 'INFO{}'.format(i)
            value = info
        # Ignore the field if there is no value.
        if value:
            result[key] = _get_value(value)

    return result


def _get_value(value):
    if not value:
        return None

    # Strip double and single quotes.
    value = value.strip('"\'')

    # Return a list if the value has a comma.
    if ',' in value:
        value = re.split(R_COMMA, value)
    # These values are equivalent to None.
    elif value in ['', '.', 'NA']:
        return None

    return value

df = dataframe(wddata+'gencode.v25.annotation.gtf')

geneMask = df['feature']=='gene'
dfGenes = df[geneMask]

GeneMask = np.isin(dfGenes['gene_name'],rnaGenes)
dfRNAgenes = dfGenes[GeneMask]

#names,counts = np.unique(dfRNAgenes['gene_name'],return_counts=True)
#badGenes = names[counts>1]
#bad = np.array(np.isin(dfRNAgenes['gene_name'],badGenes) ,dtype=bool)
#dfBad = dfRNAgenes[bad]

# dictionary geneName --> geneType
nameTypeDict = {}
for igene in range(len(dfRNAgenes)):
	nameTypeDict[ dfRNAgenes['gene_name'].iloc[igene] ] = dfRNAgenes['gene_type'].iloc[igene]

proteinCodingMask = np.ones(shape = (len(rnaGenes)),dtype=bool)
for igene in range(len(rnaGenes)):
	name = rnaGenes[igene]
	proteinCodingMask[igene] = nameTypeDict[name]=='protein_coding'
proteinCodingMask = np.array(1-proteinCodingMask,dtype=bool) # 1=bad, 0=good

np.save(wdvars+'proteinCodingMask.npy',proteinCodingMask)


