"""
Dream_code_train.py prepares data for training, trains model on test data and 
finally applies it to test data. 

Takes as input:
    TF name
    Test data to use (L, for ladder or F, for final" 

Usage:
    python Dream_code_train.py <TF-name> <stage: F, for final or L for ladderboard>
    eg: python Dream_code_train.py ATF2 L

"""

##Pure python
from multiprocessing import Pool, cpu_count
import subprocess
import pandas as pd
import numpy as np
from math import  exp
import seaborn as sns
from IPython import get_ipython

#import os moduls
import os
import sys


## Bioconda
import pybedtools
import pyBigWig
import pysam

#Biopython
from Bio.SeqUtils import GC
#get_ipython().magic(u'matplotlib inline')


# In[51]:
import matplotlib
import matplotlib.pyplot as plt


if len(sys.argv) < 3:
    print __doc__
    sys.exit(1)
BASE_DIR =os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('Dream_code_final.py'))))

# #### Determine number of Cores for parallel execution

# In[4]:

num_partitions = cpu_count()

num_cores = cpu_count()

tfs =sys.argv[1].upper()
stage = sys.argv[2].upper()

# We are performing little checks here
if stage=="L":
    final = False
elif stage == "F":
    final = True
else:
    # Exit and provide the running instructions
    print "Test data symbol not recognized"
    print __doc__
    sys.exit(1)


#Get the details of the files into a dictionarry:
a = """TF Name	Training Cell Types	Leaderboard Cell Types	Final Submission Cell Types
ARID3A	HepG2	K562	
ATF2	GM12878, H1-hESC, MCF-7	K562	HepG2
ATF3	HCT116, H1-hESC, HepG2, K562	liver	
ATF7	GM12878, HepG2, K562	MCF-7	
CEBPB	A549, H1-hESC, HCT116, HeLa-S3, HepG2, IMR-90, K562	MCF-7	
CREB1	GM12878, H1-hESC, HepG2, K562	MCF-7	
CTCF	A549, H1-hESC, HeLa-S3, HepG2, IMR-90, K562, MCF-7	GM12878	PC-3, induced_pluripotent_stem_cell
E2F1	GM12878, HeLa-S3		K562
E2F6	A549, H1-hESC, HeLa-S3	K562	
EGR1	GM12878, H1-hESC, HCT116, MCF-7	K562	liver
EP300	GM12878, H1-hESC, HeLa-S3, HepG2, K562, SK-N-SH	MCF-7	
FOXA1	HepG2	MCF-7	liver
FOXA2	HepG2		liver
GABPA	GM12878, H1-hESC, HeLa-S3, HepG2, MCF-7, SK-N-SH	K562	liver
GATA3	A549, SK-N-SH	MCF-7	
HNF4A	HepG2		liver
JUND	HCT116, HeLa-S3, HepG2, K562, MCF-7, SK-N-SH	H1-hESC	liver
MAFK	GM12878, H1-hESC, HeLa-S3, HepG2, IMR-90	K562, MCF-7	
MAX	A549, GM12878, H1-hESC, HCT116, HeLa-S3, HepG2, K562, SK-N-SH	MCF-7	liver
MYC	A549, HeLa-S3, K562, MCF-7	HepG2	
NANOG	H1-hESC		induced_pluripotent_stem_cell
REST	H1-hESC, HeLa-S3, HepG2, MCF-7, Panc1, SK-N-SH	K562	liver
RFX5	GM12878, HeLa-S3, MCF-7, SK-N-SH	HepG2	
SPI1	GM12878	K562	
SRF	GM12878, H1-hESC, HCT116, HepG2, K562	MCF-7	
STAT3	HeLa-S3	GM12878	
TAF1	GM12878, H1-hESC, HeLa-S3, K562, SK-N-SH	HepG2	liver
TCF12	GM12878, H1-hESC, MCF-7, SK-N-SH	K562	
TCF7L2	HCT116, HeLa-S3, Panc1	MCF-7	
TEAD4	A549, H1-hESC, HCT116, HepG2, K562, SK-N-SH	MCF-7	
YY1	GM12878, H1-hESC, HCT116, HepG2, SK-N-SH	K562	
ZNF143	GM12878, H1-hESC, HeLa-S3, HepG2	K562	
""".split("\n")

TF_dict = {}
for i in a:
    line = i.split("\t")
    TF_dict[line[0]] = line[1:]

factor_names = """
    ARID3A ATF2 ATF3 ATF7 CEBPB CREB1 CTCF E2F1 E2F6 EGR1 EP300 FOXA1 FOXA2 GABPA
    GATA3 HNF4A JUND MAFK MAX MYC NANOG REST RFX5 SPI1 SRF STAT3 TAF1 TCF12 TCF7L2
    TEAD4 YY1 ZNF143
    """    
# In[5]:

try:
    TF_dict[tfs]
except KeyError:
    # Exit and provide a list of accepted Tfs
    print "Input Transfription factor not recognized. Should be one of these:"
    print factor_names
    
    print __doc__
    sys.exit(1)

def energyscore(pwm_dictionary, seq):
    """
    Score sequences using the beeml energy scoring approach.

    Borrowed greatly from the work of Zhao and Stormo

    P(Si)=1/(1+e^Ei-u)

    Ei=sumsum(Si(b,k)e(b,k))

    Previous approaches seem to be using the the minimum sum of the
    energy contribution of each of the bases of a specific region.

    This is currently showing some promise but further testing is
    needed to ensure that I have a robust algorithm.
    """
    
    energy_list = []
    pwm_length = len(pwm_dictionary["A"])
    pwm_dictionary_rc = rc_pwm(pwm_dictionary, pwm_length)
    for i in range(len(seq) - 1):
        energy = 0
        energy_rc = 0
        for j in range(pwm_length - 1):
            if (j + i) >= len(seq):
                energy += 0.25
                energy_rc += 0.25
            else:
                energy += pwm_dictionary[seq[j + i]][j]
                energy_rc += pwm_dictionary_rc[seq[j + i]][j]

            energy_list.append(1 / (1 + (exp(energy))))
            energy_list.append(1 / (1 + (exp(energy_rc))))
    energy_score = min(energy_list)
    return energy_score


def rc_pwm(area_pwm, pwm_len):
    """
    Takes as input the forward pwm and returns a reverse
    complement of the motif
    """

    rcareapwm = {}
    rcareapwm["A"] = []
    rcareapwm["C"] = []
    rcareapwm["G"] = []
    rcareapwm["T"] = []
    rcareapwm["N"] = []
    for i in range(pwm_len):
        rcareapwm["A"].append(area_pwm["T"][pwm_len - i - 1])
        rcareapwm["C"].append(area_pwm["G"][pwm_len - i - 1])
        rcareapwm["G"].append(area_pwm["C"][pwm_len - i - 1])
        rcareapwm["T"].append(area_pwm["A"][pwm_len - i - 1])
        rcareapwm["N"].append(0.0)
    return rcareapwm


def get_motif(meme, motif="MOTIF"):
    """
    Extract a motif from meme file given a unique motif
    name and create dictionary for sequence scoring

    Default motif name is keyword MOTIF for single motif files. 
    """

    pwm_dictionary = {}
    pwm_dictionary["A"] = []
    pwm_dictionary["C"] = []
    pwm_dictionary["G"] = []
    pwm_dictionary["T"] = []
    pwm_dictionary["N"] = []
    flag = 0
    check = 0
    with open(meme, "r") as f1:
        for line in f1:
            if str(motif) in line:
                flag += 1
            if "letter-probability" in line and flag == 1:
                w = line.split(" ")[5]
                flag += 1
                continue
            if flag == 2 and int(check) < int(w):
                if line == "\n":
                    continue
                else:
                    words = line.split()
                    pwm_dictionary["A"].append(float(words[0]))
                    pwm_dictionary["C"].append(float(words[1]))
                    pwm_dictionary["G"].append(float(words[2]))
                    pwm_dictionary["T"].append(float(words[3]))
                    pwm_dictionary["N"].append(0.0)
                    check += 1
        return pwm_dictionary

    
def get_motif_details(tfs):
    """
    Given a TF name, create a three pwm dictionaries for scoring. 
    """
    
    tf_l = tfs.capitalize()

    pwm_motif = "%s/Motifs/%s.meme" % (BASE_DIR, tfs)

    tom_score = "%s/Motifs/%s.tomtom" % (BASE_DIR,tfs)
    mots = subprocess.Popen(["cut", "-f1", tom_score], stdout=subprocess.PIPE).stdout.read().split()
    #mots = get_ipython().getoutput(u'cut -f1 {tom_score}')
    mot = mots[1]
    pwm_dictionary = get_motif(pwm_motif, motif=mot)
    
    return pwm_dictionary

#Parallelize DFs
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

# Sequence scoring  in parallel  
def score_from_genome(bed_df):
    
    return bed_df.apply(lambda row: fetch_and_score_seq(row[0], row[1], row[2]), axis=1)

def fetch_and_score_seq(contig, start, end):
    genome = pysam.FastaFile('%s/annotations/hg19.genome.fa' % BASE_DIR)
    return score_function(pwm_dictionary, genome.fetch(contig, start, end).upper())


#Computing GC on the fly
def GC_from_genome(bed_df):
    
    return bed_df.apply(lambda row: fetch_and_get_gc(row[0], row[1], row[2]), axis=1)

def fetch_and_get_gc(contig, start, end):
    genome = pysam.FastaFile('%s/annotations/hg19.genome.fa' % BASE_DIR)
    return GC(genome.fetch(contig, start, end).upper())

def get_max_dnase(ch, start, end):
    """
    Extract the maximum fold enrichment from Bigwig files
    
    Keep in mind this error:
    "An error occurred while fetching values!"
    
    This error lead to incorrect results. 
    Will have to re-think way out latter
    
    SOLVED: The file cannot be accessed concerrently. 
    Should open a new file handle
    for each run. 
    """
    scores = []
    for cls in cl.split(","):
        
        bw = pyBigWig.open("%s/DNASE/fold_coverage_wiggles/DNASE.%s.fc.signal.bigwig" % (BASE_DIR,cls.strip()))
        try:
            scores.append(np.max(bw.values(ch, start, end)))
        except RuntimeError:
            scores.append(0)
    return np.mean(scores)

def apply_get_fold(bed_df):
    """
    Get max DNase fold enrichment over the whole
    dataframe using pandas apply function
    """
    test = bed_df.apply(lambda row: get_max_dnase(row[0], row[1], row[2]), axis=1)
    
    dnase_max = test.fillna(0)
    
    return dnase_max

def get_mean_shape(shape_file, ch, start, end):
    """
    Extract the maximum fold enrichment from Bigwig files
    
    Keep in mind this error:
    "An error occurred while fetching values!"
    
    This error lead to incorrect results. 
    Will have to re-think way out latter
    
    SOLVED: The file cannot be accessed concerrently. 
    Should open a new file handle
    for each run. 
    """
    bw = pyBigWig.open(shape_file)
    try:
        return np.mean(bw.values(ch, start, end))
    except RuntimeError:
        return 0

def apply_get_shape_roll(bed_df):
    """
    
    """
    shape_file = "%s/DNAShape/hg19.Roll.wig.bw" % BASE_DIR
    test = bed_df.apply(lambda row: get_mean_shape(shape_file, row[0], row[1], row[2]), axis=1)
    
    mean_shape = test.fillna(0)
    
    return mean_shape

def apply_get_shape_HelT(bed_df):
    """
    
    """
    shape_file = "%s/DNAShape/hg19.HelT.wig.bw" % BASE_DIR
    test = bed_df.apply(lambda row: get_mean_shape(shape_file, row[0], row[1], row[2]), axis=1)
    
    mean_shape = test.fillna(0)
    
    return mean_shape

def apply_get_shape_ProT(bed_df):
    """
    
    """
    shape_file = "%s/DNAShape/hg19.ProT.wig.bw" % BASE_DIR
    test = bed_df.apply(lambda row: get_mean_shape(shape_file, row[0], row[1], row[2]), axis=1)
    
    mean_shape = test.fillna(0)
    
    return mean_shape

def apply_get_shape_MGW(bed_df):
    """
    
    """
    shape_file = "%s/DNAShape/hg19.MGW.wig.bw" % BASE_DIR
    test = bed_df.apply(lambda row: get_mean_shape(shape_file, row[0], row[1], row[2]), axis=1)
    
    mean_shape = test.fillna(0)
    
    return mean_shape


def get_top_from_fisim(cluster_key, yylist, get_no):
    with open(cluster_key) as cluster:
        all_clusters = cluster.readlines()
    mot_lits = []
    for mot in yylist:
        for i in all_clusters:
            new_list = i.split()
            if mot in new_list:
                mot_lits.append(mot)
                all_clusters.remove(i)
    if len(mot_lits) >get_no:
        discover_three = mot_lits[:get_no]
    elif len(mot_lits) == 0:
        discover_three = yylist[0]
    else:
        discover_three = mot_lits
    return discover_three



def run_fisim_cluster(tfs):
    tomtom_raw = "%s/Motifs/%s.tomtom" % (BASE_DIR,tfs)
    dels = get_ipython().getoutput(u'python ../../Project/MAT_server/MARST_Suite/FISIM/kcmeans.py -fileIn {tomtom_raw} -o ../Motifs/{tfs}_cluster.txt -k 20      ')


def create_hdf5(tfs, col_names):
    hdf =pd.HDFStore('%s/Results/%s.h5' %(BASE_DIR ,tfs), mode='w')
    tsv_chunk = pd.read_table("%s/ChIPseq/labels/%s.train.labels.tsv" %(BASE_DIR, tfs), chunksize=500000, skiprows=1, names=col_names)
    for chunk in tsv_chunk:
        hdf.append('train', chunk, format='table', data_columns=True)
    hdf.close()

def get_querry(heds, label="B"):
    starter= ''
    columns = heds[:3]
    for i in heds[3:]:
        starter+='%s==%s & ' % (i,label)
    starter = starter[:-3]
    
    return columns, starter



def score_pwm_chunk(hdf_u, discover_three_fisim):
    """
    Given an iterable Dataframe, loop through it scoring the 
    sequences using the given PWM
    
    NB: The Hdf store should be opened and empty
    """
    for chunk in hdf_u:
        mot_scoredf_t = pd.DataFrame()
        for i, mot in enumerate(discover_three_fisim):
            pwm_motif = "%s/Motifs/%s.meme" % (BASE_DIR, tfs)
            pwm_dictionary = get_motif(pwm_motif, motif=mot)

            pwm_score = parallelize_dataframe(chunk, score_from_genome)
            i = i+1
            mot_scoredf_t["pwm_score%i" % i] = pwm_score
        hdf_score.append('pwm_score', mot_scoredf_t, format='table', data_columns=True)
def get_gc_chunk(hdf_u):
    for chunk in hdf_u:
        gc_score = parallelize_dataframe(chunk, GC_from_genome)
        hdf_score.append('gc_score', gc_score, format='table', data_columns=True)
    #return hdf_score

def get_dnase_max_chunk(hdf_u):
    for chunk in hdf_u:
        dnase_max = parallelize_dataframe(chunk, apply_get_fold)
        hdf_score.append('dnase_score',dnase_max , format='table', data_columns=True)

def get_roll_shape_chunk(hdf_u):
    for chunk in hdf_u:
        roll_shape = parallelize_dataframe(chunk, apply_get_shape_roll)
        hdf_score.append('roll_shape',roll_shape , format='table', data_columns=True)

def get_HelT_shape_chunk(hdf_u):
    for chunk in hdf_u:
        HelT_shape = parallelize_dataframe(chunk, apply_get_shape_HelT)
        hdf_score.append('HelT_shape',HelT_shape , format='table', data_columns=True)
        
def get_MGW_shape_chunk(hdf_u):
    for chunk in hdf_u:
        MGW_shape = parallelize_dataframe(chunk, apply_get_shape_MGW)
        hdf_score.append('MGW_shape',MGW_shape , format='table', data_columns=True)

# ##### Need to avoid effect below by moving the motifs to the right folder 
## and naming them appropriately

def get_ProT_shape_chunk(hdf_u):
    for chunk in hdf_u:
        ProT_shape = parallelize_dataframe(chunk, apply_get_shape_ProT)
        hdf_score.append('ProT_shape',ProT_shape , format='table', data_columns=True)




# ### Start full analyis here

# Get the training cell lines names
cl = TF_dict[tfs][0]


def mkdir_p(path):
    import os
    import errno

    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# Create a results folder for each TF in teh annotations file

mkdir_p("%s/annotations/%s" % (BASE_DIR,tfs))
print "Processing %s" % tfs
mkdir_p('%s/Results' % (BASE_DIR))

# Find a way of converting this to a workablae format without taking too much 
# memory. The current approach is just temporary, but may have to work with 
# hashtables or something to save on memory and space. The storge should in 
# a way that allows quick extraction as well as getting querying for the 
# right labels

# ### Get the columns names details

# For ease of querring teh HDF5 database, we have to rename the files here
chip= "%s/ChIPseq/labels/%s.train.labels.tsv" % (BASE_DIR, tfs)
heads = subprocess.Popen(["head", "-1", chip], stdout=subprocess.PIPE).stdout.read()
col_names = heads.replace("-","").split()



# Convert the data into an Hdf store
#TODO: I should test whether the file is already available to avoid repetition


create_hdf5(tfs, col_names)


# ### Get PWM details from the next stage

# These details are obtained by ranking the motifs using TOMTOM motif comparison
# For the purpose of this demonstration, we will provide all the files 

tomtom_raw = "%s/Motifs/%s.tomtom" % (BASE_DIR, tfs)

mot_list = subprocess.Popen(["cut", '-f1', tomtom_raw], stdout=subprocess.PIPE).stdout.read().split("\n")

cluster_key = "%s/Motifs/%s_cluster.txt" % (BASE_DIR, tfs)

discover_three_fisim = get_top_from_fisim(cluster_key, mot_list, 3)



def get_df(col_names, label="B"):
    #Given the column names, extract the iterator
    columns, starter = get_querry(col_names, label)
    
    return pd.read_hdf('%s/Results/%s.h5' % (BASE_DIR, tfs),'train',where=[starter], columns=columns, chunksize=500000)


# #### Store the data into an Hdf5 file

hdf_score =pd.HDFStore("%s/Results/%s_test.h5" % (BASE_DIR, tfs), mode='w')

print "Scoring postives"

#columns, starter = get_querry(col_names, "B")
hdf_b = get_df(col_names, "B")
score_function = energyscore
for chunk in hdf_b:
    mot_scoredf_t = pd.DataFrame()
    for i, mot in enumerate(discover_three_fisim):
        pwm_motif = "%s/Motifs/%s.meme" % (BASE_DIR, tfs)
        pwm_dictionary = get_motif(pwm_motif, motif=mot)

        pwm_score = parallelize_dataframe(chunk, score_from_genome)
        i = i+1
        mot_scoredf_t["pwm_score%i" % i] = pwm_score
    hdf_score.append('pwm_score', mot_scoredf_t, format='table', data_columns=True)


print "Scoring negatives"

hdf_u= get_df(col_names, "U")
score_function = energyscore
for chunk in hdf_u:
    mot_scoredf_t = pd.DataFrame()
    for i, mot in enumerate(discover_three_fisim):
        pwm_motif = "%s/Motifs/%s.meme" % (BASE_DIR,tfs)
        pwm_dictionary = get_motif(pwm_motif, motif=mot)

        pwm_score = parallelize_dataframe(chunk, score_from_genome)
        i = i+1
        mot_scoredf_t["pwm_score%i" % i] = pwm_score
    hdf_score.append('pwm_score', mot_scoredf_t, format='table', data_columns=True)
    

print "Scoring GC"
#get GC content for B
hdf_b= get_df(col_names, "B")
get_gc_chunk(hdf_b)

#Repeat the same for U
hdf_u= get_df(col_names, "U")
get_gc_chunk(hdf_u)

hdf_score = pd.HDFStore("%s/Results/%s_test.h5" % (BASE_DIR, tfs))

print "Scoring DNase"
#get max Dnase content for B
hdf_b= get_df(col_names, "B")
get_dnase_max_chunk(hdf_b)

#Repeat the same for U
hdf_u= get_df(col_names, "U")
get_dnase_max_chunk(hdf_u)


#TODO: Ensure this si removed later
#hdf_score = pd.HDFStore("%s/Results/%s_test.h5" % (BASE_DIR, tfs))

#hdf_score =pd.HDFStore("%s/Results/%s_test.h5" % (BASE_DIR, tfs), mode='w')

hdf_score = pd.HDFStore("%s/Results/%s_test.h5" % (BASE_DIR, tfs))

#hdf_score =pd.HDFStore("%s/Results/%s_test.h5" % (BASE_DIR, tfs), mode='w')

print "Scoring roll shape"

#get roll shape content for B
hdf_b= get_df(col_names, "B")
get_roll_shape_chunk(hdf_b)

hdf_b = get_df(col_names, "B")
get_HelT_shape_chunk(hdf_b)

hdf_b = get_df(col_names, "B")
get_MGW_shape_chunk(hdf_b)

hdf_b = get_df(col_names, "B")
get_ProT_shape_chunk(hdf_b)


#Repeat the same for negatives\

hdf_u= get_df(col_names, "U")
get_roll_shape_chunk(hdf_u)

hdf_u = get_df(col_names, "U")
get_HelT_shape_chunk(hdf_u)

hdf_u = get_df(col_names, "U")
get_MGW_shape_chunk(hdf_u)

hdf_u = get_df(col_names, "U")
get_ProT_shape_chunk(hdf_u)

hdf_score.close()
    

# ### Score working with the ladderboard sequences


#load the file, chunkwise
if final:
    print "Loading final data"
else:
    print "Loading ladder data"

col_name = ["contig", "start", "stop"]
hdf_lad =pd.HDFStore('%s/Results/%s_ladder.h5' % (BASE_DIR, tfs), mode='w')
tsv_chunk = pd.read_table("%s/annotations/ladder_regions.blacklistfiltered.bed" % BASE_DIR, chunksize=500000, names=col_name)
for chunk in tsv_chunk:
    hdf_lad.append('ladder', chunk, format='table', data_columns=True, min_itemsize=5)
hdf_lad.close()

"""
# ### PWM score

# In[35]:

# def score_pwm_chunk(hdf_u, discover_three_fisim):
#     
#     Given an iterable Dataframe, loop through it scoring the 
#     sequences using the given PWM
    
#     NB: The Hdf store should be opened and empty
#  
"""
## Create am HDF5 store for the results
if final:
    hdf_score =pd.HDFStore('%s/Results/%s_ladder_score.h5' % (BASE_DIR, tfs))
else:
    hdf_score =pd.HDFStore('%s/Results/%s_ladder_score.h5' % (BASE_DIR, tfs))


def read_ladder():
    """
    Uses global information to determine if we are working with final or not
    """
    if final:
        return pd.read_hdf('%s/Results/%s_final.h5' % (BASE_DIR, tfs), 'final',  chunksize=500000)
    else:
        return pd.read_hdf('%s/Results/%s_ladder.h5' % (BASE_DIR, tfs), 'ladder',  chunksize=500000)


hdf_ladder = read_ladder()

print "Getting PWM scores for ladder"
score_function = energyscore
for chunk in hdf_ladder:
    mot_scoredf_t = pd.DataFrame()
    for i, mot in enumerate(discover_three_fisim):
        pwm_motif = "%s/Motifs/%s.meme" % (BASE_DIR,tfs)
        pwm_dictionary = get_motif(pwm_motif, motif=mot)

        pwm_score = parallelize_dataframe(chunk, score_from_genome)
        i = i+1
        mot_scoredf_t["pwm_score%i" % i] = pwm_score
    hdf_score.append('pwm_score', mot_scoredf_t, format='table', data_columns=True)


# ### GC score

print "Getting GC for ladder"
hdf_ladder = read_ladder()
get_gc_chunk(hdf_ladder)

if final:
    cl = TF_dict[tfs][2].split(",")[0]
else:
    cl = TF_dict[tfs][1].split(",")[0]

print "Getting DNase for ladder"
hdf_ladder = read_ladder()
get_dnase_max_chunk(hdf_ladder)



print "Getting roll for ladder"
hdf_ladder = read_ladder()
get_roll_shape_chunk(hdf_ladder)

hdf_ladder = read_ladder()
get_HelT_shape_chunk(hdf_ladder)

hdf_ladder = read_ladder()
get_MGW_shape_chunk(hdf_ladder)

hdf_ladder = read_ladder()
get_ProT_shape_chunk(hdf_ladder)


hdf_score.close()

## For use to determine the size of positive and negative sets
hdf_b= get_df(col_names, "B")
hdf_u= get_df(col_names, "U")



# In[24]:
print "Reading training data to DF"

model_store = "%s/Results/%s_test.h5" % (BASE_DIR, tfs)
hdf_scores_pw = pd.read_hdf(model_store, key="pwm_score")
hdf_scores_pw["dnase_score"] = pd.read_hdf(model_store, key="dnase_score")
hdf_scores_pw["gc_score"] = pd.read_hdf(model_store, key="gc_score")
hdf_scores_pw["roll_shape"] = pd.read_hdf(model_store, key="roll_shape")
hdf_scores_pw["HelT_shape"] = pd.read_hdf(model_store, key="HelT_shape")
hdf_scores_pw["MGW_shape"] = pd.read_hdf(model_store, key="MGW_shape")
hdf_scores_pw["ProT_shape"] = pd.read_hdf(model_store, key="ProT_shape")

# ### Load test data

print "Loading ladder data to DF"
if final:
    ladder_store = "%s/Results/%s_final_score.h5" % (BASE_DIR, tfs)
else:
    ladder_store = "%s/Results/%s_ladder_score.h5" % (BASE_DIR, tfs)

hdf_scores_lad = pd.read_hdf(ladder_store, key="pwm_score")
hdf_scores_lad["dnase_score"] = pd.read_hdf(ladder_store, key="dnase_score")
hdf_scores_lad["gc_score"] = pd.read_hdf(ladder_store, key="gc_score")
hdf_scores_lad["roll_shape"] = pd.read_hdf(ladder_store, key="roll_shape")
hdf_scores_lad["HelT_shape"] = pd.read_hdf(ladder_store, key="HelT_shape")
hdf_scores_lad["MGW_shape"] = pd.read_hdf(ladder_store, key="MGW_shape")
hdf_scores_lad["ProT_shape"] = pd.read_hdf(ladder_store, key="ProT_shape")


neg_size = len(hdf_u.coordinates)
pos_size = len(hdf_b.coordinates)
y = np.concatenate((np.ones(pos_size), np.zeros(neg_size)), axis=0)


from sklearn.externals import joblib

import xgboost as xgb

import pickle


def train_xgboost(dataframe, y):
    """
    Given a feature DF, train a model using the optimized parameters
    
    Latter on, I will need to find a way to optimize these parameters for a different TF
    """
    xgdmat = xgb.DMatrix(dataframe, y) 

    our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth':7, 'min_child_weight':1} 

    final_gb = xgb.train(our_params, xgdmat, num_boost_round = 3000)
    
    #save the model for future reference
    #pickle.dump(final_gb, "%s/annotations/%s/%s_xgboost_pick.dat" % (BASE_DIR, tfs, tfs))
    #joblib.dump(final_gb, "%s/annotations/%s/%s_xgboost.dat" % (BASE_DIR, tfs, tfs))
    
    #Creat a feature importance plot
    #plot_feature_importance(final_gb, "%s/annotations/%s/%s_features.png" % (BASE_DIR, tfs, tfs))
    
    return final_gb


def plot_feature_importance(xgb_model, fig_out):
    sns.set(font_scale = 1.5)
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    xgb.plot_importance(xgb_model, ax=ax)
    #ax.plot([0,1,2], [10,20,3])
    fig.savefig(fig_out, bbox_inches='tight')   # save the figure to file
    plt.close(fig)


# ### Using the Xgboost model to predict on the test data

print "Training the model"

final_gb = train_xgboost(hdf_scores_pw, y)


print "Using the model to predict on the ladder data"

testdmat = xgb.DMatrix(hdf_scores_lad)
y_pred = final_gb.predict(testdmat) # Predict using our testdmat

print "Saving the model for latter reference"

plot_feature_importance(final_gb, "%s/annotations/%s/%s_features.png" % (BASE_DIR, tfs, tfs))
np.savetxt("%s/annotations/%s/%s_xgb_v2.txt" % (BASE_DIR, tfs,tfs), y_pred)

joblib.dump(final_gb, "%s/annotations/%s/%s_xgboost_v2.dat" % (BASE_DIR, tfs, tfs))

# combine the data with respective data, zip and submit for scoring
