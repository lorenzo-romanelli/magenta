import csv
import random
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from itertools import islice

'''
The following dictionary contains the metadata of the datasets used later in 
the code, namely:
    
    - the path to the dataset
    - the number of records of the dataset
    - the prefix to identify the dataset

NOTE: you can get the number of records in a CSV file using the following
bash command (on Linux):
       
    $ wc -l <filename>
 
Counting the records of the CSV using Python directly is not feasible,
because we would need to read the file twice.
'''
datasets_meta = {
    'allmusic': {
        'path': './datasets/acousticbrainz-mediaeval2017-allmusic-train.tsv',
        'nrecords': 1353214,
        'prefix': 'a'
    },
    'discogs': {
        'path': './datasets/acousticbrainz-mediaeval2017-discogs-train.tsv',
        'nrecords': 904945,
        'prefix': 'd'
    },
    'lastfm': {
        'path': './datasets/acousticbrainz-mediaeval2017-lastfm-train.tsv',
        'nrecords': 566711,
        'prefix': 'l'
    },
    'tagtraum': {
        'path': './datasets/acousticbrainz-mediaeval2017-tagtraum-train.tsv',
        'nrecords': 486741,
        'prefix': 't'
    }
}

genres = dict()
subgenres = dict()
rec_mbids = list()

def isGenre(s):
    '''
    Checks if the string <s> is a genre label.
    '''
    return len(s.split('---')) == 1

def isSubGenre(s):
    '''
    Checks if the string <s> is a subgenre label.
    '''
    return len(s.split('---')) == 2

def plotDendrogram(Z, labels, output_file = None):
    '''
    Plot the dendrogram associated with the clustered matrix <Z>,
    and save the plot in <output_file> (if specified).
    '''
    plt.figure(figsize = (25, 30))
    dendrogram(Z, labels = labels, orientation = 'left')
    plt.plot()
    # plt.tight_layout()
    if output_file != None:
        plt.gcf()
        plt.savefig(output_file)
    plt.show()

def getGenresAndSubgenres(labels):
    '''
    Splits labels in two subsets: genres and subgenres.
    Subgenres are always in the form "<genre>---<subgenre>".
    Returns labels for genres and subgenres.
    '''
    true_labels = [label for label in labels if label != '']      # get rid of empty labels
    genre_labels = [label for label in true_labels if isGenre(label)]
    subgenre_labels = [label for label in true_labels if isSubGenre(label)]
    return (genre_labels, subgenre_labels)

def buildDictionary(dictionary, labels, mbid):
    '''
    Builds (or updates) the dictionary containing subgenres (or genres)
    associated with the MusicBrainz's id of the tracks corresponding 
    to that subgenre (or genre).
    '''
    for label in labels:
        if dictionary.get(label) == None:
            dictionary[label] = []
        dictionary[label].append(mbid)

def createBagOfWords(subgenre_dictionary, threshold = None):
    '''
    Builds the bag-of-words for all the specified subgenres, only considering
    the ones that have a number of associated tracks equal or greater than
    a threshold (if specified).
    Returns the labels of the subgenres taken into consideration and the
    corresponding bag-of-words.
    '''
    if threshold is not None:
        sg_dict_filt = {sg: tracks for sg, tracks in subgenre_dictionary.items() if len(tracks) > threshold}
    else:
        sg_dict_filt = subgenre_dictionary
    mx = [[1 if track in set(tracks) else 0 for track in rec_mbids] for tracks in sg_dict_filt.values()]
    labels = list(sg_dict_filt.keys())
    return (labels, mx)

def readCsvDataset(dataset, delimiter = ' ', start = 0, nitems = None):
    '''
    Reads the specified CSV dataset, starting from the element specified by
    the <start> parameter (useful for skipping CSV headers).
    <nitems> random elements are read from the dataset, processed and stored into
    the dictionaries containing genres and subgenres associations with tracks.
    The <nrecords> parameter is needed in order to know how many elements are stored 
    into the CSV dataset (see comment related to the <records_number> variable at
    the top of this file). 
    '''
    nrecords = dataset['nrecords']
    path_to_dataset = dataset['path']
    prefix = dataset['prefix']
    with open(path_to_dataset) as d:
        reader = csv.reader(d, delimiter = delimiter)
        rand_idxs = random.sample(range(start, nrecords), nitems) if nitems is not None else range(start, nrecords)
        rand_idxs = set(rand_idxs)
        reader = csv.reader(d, delimiter = delimiter)
        for idx, line in enumerate(islice(reader, start, None)):
            if idx not in rand_idxs:
                continue
            rec_mbid = line[0]
            genre_labels, subgenre_labels = getGenresAndSubgenres(line[2:])
            # Uncomment next line if you're interested in storing genre labels
            # buildDictionary(dictionary = genres, labels = genre_labels, mbid = rec_mbid)
            subgenre_labels = [prefix + '::' + sgl.split('---')[1] for sgl in subgenre_labels]
            buildDictionary(dictionary = subgenres, labels = subgenre_labels, mbid = rec_mbid)
            rec_mbids.append(rec_mbid)

if __name__ == '__main__':
    start = 1           # skip first line of the csv (header)
    nitems = 20000      # only work on n random records (if None, work on everything)
    threshold = 200     # only work on subgenres that havea at least n recordings associated 
                        # (if None, work on everything)
    out_filename = ""
    # datasets to be used for the analysis
    # (uncomment the datasets you want to analyze)
    datasets = [
        #'tagtraum',
        #'lastfm',
        #'discogs',
        #'allmusic',
    ]
    for dataset in datasets:
        readCsvDataset(dataset = datasets_meta[dataset], delimiter = '\t',
                       start = start, nitems = nitems)
        out_filename += datasets_meta[dataset]['prefix']
    labels, X = createBagOfWords(subgenres, threshold)
    Y = pdist(X, 'cosine')
    Z = linkage(Y)
    # Plots are saved following this scheme:
    #       d_r_t.png
    # where
    #   - d are the datasets analyzed, identified by their prefix,
    #   - r is the number of records considered,
    #   - t is the minimum number of records associated with each subgenre.
    nit = "all" if nitems is None else str(nitems)
    th = "all" if threshold is None else str(threshold)
    output_file = "./images/" + out_filename + "_" + nit + "_" + th + ".png"
    plotDendrogram(Z, labels, output_file)
