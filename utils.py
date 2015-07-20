def tf(tokens):
    """ Compute TF
    Args:
        tokens (list of str): input list of tokens from tokenize
    Returns:
        dictionary: a dictionary of tokens to its TF values
    """
    token_dict = dict()   
    for token in tokens:
        if token in token_dict:
            token_dict[token] = token_dict[token] + 1
        else:
            token_dict[token] = 1
            
    for t in token_dict:
        token_dict[t] = float(token_dict[t])/float(len(tokens))
        
    return token_dict

def idfs(corpus):
    """ Compute IDF
    Args:
        corpus (RDD): input corpus
    Returns:
        RDD: a RDD of (token, IDF value)
    """
    N = corpus.count()
    uniqueTokens = corpus.flatMap(lambda doc: set(doc[1]))
    tokenCountPairTuple = uniqueTokens.map(lambda t: (t, 1))
    tokenSumPairTuple = tokenCountPairTuple.reduceByKey(lambda a,b: a+b)
    return tokenSumPairTuple.map(lambda (k, v): (k, math.log(1.0*N/v)))    


def tfidf(tokens, idfs):
    """ Compute TF-IDF
    Args:
        tokens (list of str): input list of tokens from tokenize
        idfs (dictionary): record to IDF value
    Returns:
        dictionary: a dictionary of records to TF-IDF values
    """
    tfs = tf(tokens)
    tfIdfDict = {k: v*idfs[k] for k, v in tfs.items()}
    return tfIdfDict


def dotprod(a, b):
    """ Compute dot product
    Args:
        a (dictionary): first dictionary of record to value
        b (dictionary): second dictionary of record to value
    Returns:
        dotProd: result of the dot product with the two input dictionaries
    """
    dp=0
    for k in a:
        if k in b:
            dp += a[k] * b[k]
    return  dp

def norm(a):
    """ Compute square root of the dot product
    Args:
        a (dictionary): a dictionary of record to value
    Returns:
        norm: a dictionary of tokens to its TF values
    """
    return math.sqrt(dotprod(a, a))

def cosineSimilarity(record, idfsRDD, idfsRDD2, corpusNorms1, corpusNorms2):
    """ Compute Cosine Similarity using Broadcast variables
    Args:
        record: ((ID1, ID2), token), RDDs (Broadcast) with idfs and norm values
    Returns:
        pair: ((ID1, ID2), cosine similarity value)
    """
    vect1Rec = record[0][0]
    vect2Rec = record[0][1]
    tokens = record[1]
    s = sum((idfsRDD[vect1Rec][i]*idfsRDD2[vect2Rec][i] for i in tokens))
    value = s/((corpusNorms1[vect1Rec])*(corpusNorms2[vect2Rec]))
    key = (vect1Rec, vect2Rec)
    return (key, value)