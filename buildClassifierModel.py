#!/usr/bin/env python3

import sys
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report

INDEXING = 1 # typically, sequence indices either start at 1 or zero #

## parameters controlling the testing and cross-validation      ##
OS_TEST_COUNT     = 10  # number of out-of-sample tests to do    #
OS_TEST_SIZE_PROP = 0.4 # out-of-sample test proportion          #
CV_FOLD           = 5   # fold cross-validation to use           #

datadir = "data"
binding_pocket_fname = "%s/bindingPocketCols_1index.txt" % datadir
labelled_data_fname  = "%s/labelled.csv"                 % datadir
alignment_fname      = "%s/allSeqsAligned.fasta"         % datadir
asr_prob_dirname     = "%s/asr_post_distns"              % datadir


#### beg function definitions ####

def parseFasta(fname):
    result = {}
    handle = open(fname, "r")
    line   = handle.readline()
    while line:
        if line[0] == ">":
            id = line[1:].strip()
            se = ""
            line = handle.readline()
            while line and line[0] != ">":
                se += line.strip()
                line = handle.readline()
            result[id] = se.upper()
    handle.close()
    return result

# amino-acid substitutions from JTT #
#   P(0.01), amino acid exchange data generated from SWISSPROT Release 22.0
#   Ref. Jones D.T., Taylor W.R. and Thornton J.M. (1992) CABIOS 8:275-282
amino_acid_subst_matrix = [
[0.98754, 0.00030, 0.00023, 0.00042, 0.00011, 0.00023, 0.00065, 0.00130, 0.00006, 0.00020, 0.00028, 0.00021, 0.00013, 0.00006, 0.00098, 0.00257, 0.00275, 0.00001, 0.00003, 0.00194],
[0.00044, 0.98974, 0.00019, 0.00008, 0.00022, 0.00125, 0.00018, 0.00099, 0.00075, 0.00012, 0.00035, 0.00376, 0.00010, 0.00002, 0.00037, 0.00069, 0.00037, 0.00018, 0.00006, 0.00012],
[0.00042, 0.00023, 0.98720, 0.00269, 0.00007, 0.00035, 0.00036, 0.00059, 0.00089, 0.00025, 0.00011, 0.00153, 0.00007, 0.00004, 0.00008, 0.00342, 0.00135, 0.00001, 0.00022, 0.00011],
[0.00062, 0.00008, 0.00223, 0.98954, 0.00002, 0.00020, 0.00470, 0.00095, 0.00025, 0.00006, 0.00006, 0.00015, 0.00004, 0.00002, 0.00008, 0.00041, 0.00023, 0.00001, 0.00015, 0.00020],
[0.00043, 0.00058, 0.00015, 0.00005, 0.99432, 0.00004, 0.00003, 0.00043, 0.00016, 0.00009, 0.00021, 0.00004, 0.00007, 0.00031, 0.00007, 0.00152, 0.00025, 0.00016, 0.00067, 0.00041],
[0.00044, 0.00159, 0.00037, 0.00025, 0.00002, 0.98955, 0.00198, 0.00019, 0.00136, 0.00005, 0.00066, 0.00170, 0.00010, 0.00002, 0.00083, 0.00037, 0.00030, 0.00003, 0.00008, 0.00013],
[0.00080, 0.00015, 0.00025, 0.00392, 0.00001, 0.00130, 0.99055, 0.00087, 0.00006, 0.00006, 0.00009, 0.00105, 0.00004, 0.00002, 0.00009, 0.00021, 0.00019, 0.00001, 0.00002, 0.00029],
[0.00136, 0.00070, 0.00035, 0.00067, 0.00012, 0.00011, 0.00074, 0.99350, 0.00005, 0.00003, 0.00006, 0.00016, 0.00003, 0.00002, 0.00013, 0.00137, 0.00020, 0.00008, 0.00003, 0.00031],
[0.00021, 0.00168, 0.00165, 0.00057, 0.00014, 0.00241, 0.00016, 0.00017, 0.98864, 0.00009, 0.00051, 0.00027, 0.00008, 0.00016, 0.00058, 0.00050, 0.00027, 0.00001, 0.00182, 0.00008],
[0.00029, 0.00011, 0.00020, 0.00006, 0.00003, 0.00004, 0.00007, 0.00004, 0.00004, 0.98729, 0.00209, 0.00012, 0.00113, 0.00035, 0.00005, 0.00027, 0.00142, 0.00001, 0.00010, 0.00627],
[0.00023, 0.00019, 0.00005, 0.00004, 0.00005, 0.00029, 0.00006, 0.00005, 0.00013, 0.00122, 0.99330, 0.00008, 0.00092, 0.00099, 0.00052, 0.00040, 0.00015, 0.00007, 0.00008, 0.00118],
[0.00027, 0.00331, 0.00111, 0.00014, 0.00001, 0.00118, 0.00111, 0.00020, 0.00011, 0.00011, 0.00013, 0.99100, 0.00015, 0.00002, 0.00011, 0.00032, 0.00060, 0.00001, 0.00003, 0.00009],
[0.00042, 0.00023, 0.00013, 0.00008, 0.00006, 0.00018, 0.00011, 0.00011, 0.00007, 0.00255, 0.00354, 0.00038, 0.98818, 0.00017, 0.00008, 0.00020, 0.00131, 0.00003, 0.00006, 0.00212],
[0.00011, 0.00003, 0.00004, 0.00002, 0.00015, 0.00002, 0.00003, 0.00004, 0.00009, 0.00047, 0.00227, 0.00002, 0.00010, 0.99360, 0.00009, 0.00063, 0.00007, 0.00008, 0.00171, 0.00041],
[0.00148, 0.00038, 0.00007, 0.00008, 0.00003, 0.00067, 0.00011, 0.00018, 0.00026, 0.00006, 0.00093, 0.00012, 0.00004, 0.00007, 0.99270, 0.00194, 0.00069, 0.00001, 0.00003, 0.00015],
[0.00287, 0.00052, 0.00212, 0.00031, 0.00044, 0.00022, 0.00018, 0.00146, 0.00017, 0.00021, 0.00054, 0.00027, 0.00007, 0.00037, 0.00144, 0.98556, 0.00276, 0.00005, 0.00020, 0.00025],
[0.00360, 0.00033, 0.00098, 0.00020, 0.00008, 0.00021, 0.00020, 0.00024, 0.00011, 0.00131, 0.00024, 0.00060, 0.00053, 0.00005, 0.00060, 0.00324, 0.98665, 0.00002, 0.00007, 0.00074],
[0.00007, 0.00065, 0.00003, 0.00002, 0.00023, 0.00008, 0.00006, 0.00040, 0.00002, 0.00005, 0.00048, 0.00006, 0.00006, 0.00021, 0.00003, 0.00024, 0.00007, 0.99686, 0.00023, 0.00017],
[0.00008, 0.00010, 0.00030, 0.00024, 0.00041, 0.00010, 0.00004, 0.00006, 0.00130, 0.00017, 0.00022, 0.00005, 0.00004, 0.00214, 0.00005, 0.00043, 0.00012, 0.00010, 0.99392, 0.00011],
[0.00226, 0.00009, 0.00007, 0.00016, 0.00012, 0.00008, 0.00027, 0.00034, 0.00003, 0.00511, 0.00165, 0.00008, 0.00076, 0.00025, 0.00012, 0.00026, 0.00066, 0.00004, 0.00005, 0.98761]
]

# background amino-acid match state frequencies from UCSC SAM 3.5's default
# Dirichlet mixture regularizer, rsdb-comp2.32comp
amino_acids = ['A',    'R',    'N',    'D',    'C',    'E',    'Q',    'G',    'H',    'I',    'L',    'K',    'M',    'F',    'P',    'S',    'T',    'W',    'Y',    'V'    ]
backgroundp = [0.08713,0.04090,0.04043,0.04687,0.03347,0.04953,0.03826,0.08861,0.03362,0.03689,0.08536,0.08048,0.01475,0.03977,0.05068,0.06958,0.05854,0.01049,0.02992,0.06472]
#backgroundp = [0.084,  0.049,  0.045,  0.048,  0.017,  0.059,  0.041,  0.057,  0.023,  0.071,  0.090,  0.059,  0.030,  0.042,  0.032,  0.070,  0.062,  0.010,  0.032,  0.082  ]

def getProbDist(aa, other_seqs):
    result = []
    counts = backgroundp

    # get probability distribution for my sequence #
    counts = [ sum(x) for x in zip(counts, amino_acid_subst_matrix[amino_acids.index(aa)]) ]
    for other_aa in other_seqs:
        [ sum(x) for x in zip(counts, amino_acid_subst_matrix[amino_acids.index(other_aa)]) ]

    finalresult = [x/sum(counts) for x in counts]
    return finalresult

def buildSequenceData(ids, sequence_data, pocket_cols, labels):
    result = []
    for idndx in range(len(ids)):
        id       = ids[idndx]
        label    = labels[idndx]
        sequence = sequence_data[id]
        mydata   = []
        for col in pocket_cols:
            aa = sequence[col]
            other_seqs = []
            for i in range(len(ids)):
                if labels[i] == label:
                    other_seqs.append(sequence_data[ids[i]][col])
            mydata.extend(getProbDist(aa, other_seqs))
        result.append(mydata)
    return np.array(result)

def parseASRProbDist(pdistfname, pocket_cols):
    result = []
    pdist  = {}
    handle = open(pdistfname, "r")
    for line in handle:
        linearr = line.split()
        col     = int(linearr[0])-INDEXING
        ppdist  = {}
        for i in range(1,len(linearr),2):
            aa = linearr[i]
            pp = float(linearr[i+1][1:-1])
            ppdist[aa] = pp
        pdist[col] = ppdist
    handle.close()
    for column in pocket_cols:
        my_results = [0.0]*len(amino_acids)
        if column in pdist.keys():
            mypdist = pdist[column]
            for i in range(len(amino_acids)):
                amino_acid = amino_acids[i]
                if amino_acid in mypdist.keys():
                    my_results[i] = mypdist[amino_acid]
        result.extend(my_results)
    return result

def buildPredictData(sequence_data, pocket_cols):
    result = []
    ## build data for ML sequences ##
    ids = list(sequence_data.keys())
    for id in ids:
        mydata = []
        for col in pocket_cols:
            aa = sequence_data[id][col]
            probdist = [0.0] * len(amino_acids)
            if aa in amino_acids:
                probdist[amino_acids.index(aa)] = 1.0
            mydata.extend(probdist)
        result.append(mydata)
    ## build data from ASR probability distributions ##
    for pdistfname in glob.glob("%s/*.txt" % asr_prob_dirname):
        myid = "ASRPROBDIST_" + pdistfname.split("/")[-1].split(".txt")[0]
        mydata = parseASRProbDist(pdistfname, pocket_cols)
        ids.append(myid)
        result.append(mydata)
    return (ids, np.array(result))

#### end function definitions ####


### read binding pocket columns ###
binding_pocket_cols = []
handle = open(binding_pocket_fname, "r")
for line in handle:
    binding_pocket_cols.append( int(line) - INDEXING )
handle.close()
binding_pocket_cols.sort()

### read labels for trainig data ###
ids    = []
labels = []
handle = open(labelled_data_fname, "r")
for line in handle:
    linearr = line.strip().split(",")
    ids.append(linearr[0])
    labels.append(linearr[1])
handle.close()
all_labelled_ids = np.array(ids)
all_labels       = np.array(labels)

### read all aligned sequence data ###
all_sequence_data = parseFasta(alignment_fname)

### convert alignment to training data ###
all_data = buildSequenceData(all_labelled_ids, all_sequence_data, binding_pocket_cols, all_labels)

### convert alignment to data for prediction ###
(all_ids, all_predict_data) = buildPredictData(all_sequence_data, binding_pocket_cols)

### plot PCA ###
n_components = 2
pca = PCA(n_components=n_components)
X_transformed = pca.fit_transform(all_data)
colors = ['navy', 'turquoise', 'darkorange']

title = "PCA"
plt.figure(figsize=(4, 4))
for color, target_name in zip(colors, sorted(list(set(all_labels)))):
    plt.scatter(X_transformed[all_labels == target_name, 0], X_transformed[all_labels == target_name, 1],
                color=color, lw=2, label=target_name)

plt.title("PCA")
plt.legend(loc="best", shadow=False, scatterpoints=1)
#plt.show()

### set up training grid parameters ###
train_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

### split labelled training data into training and testing sets ###
main_shuffle_split = StratifiedShuffleSplit(n_splits=OS_TEST_COUNT,
                                            test_size=OS_TEST_SIZE_PROP,
                                            random_state=0)
testround = 1
for train_indices, test_indices in main_shuffle_split.split(all_labelled_ids, all_labels):
    print("TESTING ROUND %d" % testround)
    testround += 1

    train_ids    = all_labelled_ids[train_indices]
    train_labels = all_labels[train_indices]
    #train_data   = all_data[train_indices]
    train_data   = buildSequenceData(train_ids, all_sequence_data, binding_pocket_cols, train_labels)

    test_ids     = all_labelled_ids[test_indices]
    test_labels  = all_labels[test_indices]
    #test_data    = all_data[test_indices]
    test_data   = buildSequenceData(test_ids, all_sequence_data, binding_pocket_cols, test_labels)

    ### grid search with CV for model selection ###
    clf = GridSearchCV(SVC(probability=False, max_iter=50000, decision_function_shape='ovr', gamma='scale', class_weight='balanced'), train_grid, cv=CV_FOLD)
    clf.fit(train_data, train_labels)

    ### get scores and stuff ###
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    ### prediction test ###
    predicted_test_labels = clf.predict(test_data)
    print(classification_report(test_labels, predicted_test_labels))

### done training; build the 'real' model and classify the unknowns ###
svc = clf.best_estimator_
all_predicted_support = svc.decision_function(all_predict_data)
all_predicted_labels  = svc.predict(all_predict_data)
sys.stdout.write("id,predicted_label")
the_labels = sorted(list(set(all_labels)))
for the_label in the_labels:
    sys.stdout.write(",%s" % the_label)
sys.stdout.write("\n")
for i in range(len(all_ids)):
    sys.stdout.write(all_ids[i])
    sys.stdout.write(",%s" % all_predicted_labels[i])
    for score in all_predicted_support[i]:
        sys.stdout.write(",%.4f" % score)
    sys.stdout.write("\n")
