import sys, os, math
import re, unicodedata
import nltk
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
from pymongo import MongoClient
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import Vectors, SparseVector
from pyspark.mllib.regression import LabeledPoint

#general variables
#MongoDB
host = '192.168.33.10'
port = 27017
username = ''
password = ''
database = 'recsysdb'

APP_NAME = 'Recomender System'

#connecting to MongoDB
def createMongoDBConnection(host, port, username, password, db):
	""" create connection with MongoDB
    Args:
        params to connection
    Returns:
        a connection to db
    """
	client = MongoClient(host, port)
	return client[db]

def findProductsByCategory(categories):
	""" get products on MongoDB
    Args:
        categories (list): input categories to search
    Returns:
        list: a list of the products
    """
	db = createMongoDBConnection(host, port, username, password, database)
	produtos = db.produto
	product_list = []

	for produto in produtos.find({"categorias.nome" : {"$in" : categories}}):
		keys = produto.keys()
		
		description = ''
		if 'descricaoLonga' in keys:
			description = description + produto['descricaoLonga']

		if 'nome' in keys:
			description = description + produto ['nome']
		
		id = None
		if '_id' in keys:
			id = produto['_id']
		
		category = None
		if 'categorias' in keys:
			category = produto['categorias'][2]['nome']

		product_list.append((id, description, category))

	return product_list
                   

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


def main(sc):
    categs1 = ["Computers & Tablets", "Video Games", "TV & Home Theater"]#, "Musical Instruments"]
    categs2 = ["Computers & Tablets"]

    stpwrds = stopwords.words('english')
    tbl_translate = dict.fromkeys(i for i in xrange(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P') or unicodedata.category(unichr(i)).startswith('N'))
    
    #Set 1
    productRDD = sc.parallelize(findProductsByCategory(categs1))

    categories = productRDD.map(lambda x: x[2]).distinct().collect()

    corpusRDD = (productRDD.map(lambda s: (s[0], word_tokenize(s[1].translate(tbl_translate).lower()), s[2]))
						   .map(lambda s: (s[0], [PorterStemmer().stem(x) for x in s[1] if x not in stpwrds], categories.index(s[2]))))

    tokens = corpusRDD.flatMap(lambda x: x[1]).distinct().collect()
    numTokens = len(tokens)

    idfsRDD = idfs(corpusRDD)
    idfsRDDBroadcast = sc.broadcast(idfsRDD.collectAsMap())
    tfidfRDD = corpusRDD.map(lambda x: (x[0], tfidf(x[1], idfsRDDBroadcast.value), x[2]))
    tfidfRDDBroadcast = sc.broadcast(tfidfRDD.map(lambda x: (x[0], x[1])).collectAsMap())
    corpusInvPairsRDD = tfidfRDD.flatMap(lambda r: ([(x, r[0]) for x in r[1]])).cache()

    vectSpaceRDD1 = tfidfRDD.map(lambda t: LabeledPoint(t[2], SparseVector(numTokens, sorted([tokens.index(i) for i in t[1].keys()]), [t[1][tokens[i]] for i in sorted([tokens.index(i) for i in t[1].keys()])])))
    
    trainingRDD, testRDD = vectSpaceRDD1.randomSplit([8, 2], seed=0L)

    model = NaiveBayes.train(trainingRDD)
    predictionAndLabel = testRDD.map(lambda p : (model.predict(p.features), p.label))

    acuraccy = float(predictionAndLabel.filter(lambda (x, v): x == v).count())/float(predictionAndLabel.count())

    print predictionAndLabel.take(10)
    print acuraccy

    #Set 2
    #productRDD2 = sc.parallelize(findProductsByCategory(categs2))
    #corpusRDD2 = (productRDD2.map(lambda s: (s[0], word_tokenize(s[1].translate(tbl_translate).lower())))
                             #.map(lambda s: (s[0], [PorterStemmer().stem(x) for x in s[1] if x not in stpwrds])))

    #idfsRDD2 = idfs(corpusRDD2)
    #idfsRDDBroadcast2 = sc.broadcast(idfsRDD2.collectAsMap())
    #tfidfRDD2 = corpusRDD2.map(lambda            x: (x[0], tfidf(x[1], idfsRDDBroadcast2.value)))
    #tfidfRDDBroadcast2 = sc.broadcast(tfidfRDD2.collectAsMap())
    #produtosTeste = sc.parallelize(tfidfRDD2.take(3))
    #corpusInvPairsRDD2 = produtosTeste.flatMap( lambda r: ([(x, r[0]) for x in r[1]])).cache()

    #vectSpaceRDD2 = tfidfRDD2.map(lambda t: LabeledPoint(t[0], SparseVector(numTokens, sorted([tokens.index(i) for i in t[1].keys()]), [t[1][tokens[i]] for i in sorted([tokens.index(i) for i in t[1].keys()])]))) 

    #commonTokens = (corpusInvPairsRDD1.join(corpusInvPairsRDD2)
                                      #.map(lambda x: (x[1], x[0]))
                                      #.groupByKey()
                                      #.cache())

    #corpusNorms1 = tfidfRDD1.map(lambda x: (x[0], norm(x[1])))
    #corpusNormsBroadcast1 = sc.broadcast(corpusNorms1.collectAsMap())
    #corpusNorms2 = tfidfRDD2.map(lambda x: (x[0], norm(x[1])))
    #corpusNormsBroadcast2 = sc.broadcast(corpusNorms2.collectAsMap())

    #similaritiesRDD =  (commonTokens.map(lambda x: cosineSimilarity(x, tfidfRDDBroadcast1.value, tfidfRDDBroadcast2.value, corpusNormsBroadcast1.value, corpusNormsBroadcast2.value)).cache())
    
    #print similaritiesRDD.filter(lambda x: x[0][0]==u'1051384145329' and x[0][1]==u'1051384145329').collect()
    
if __name__ == '__main__':
	conf = SparkConf().setAppName(APP_NAME)
	sc = SparkContext(conf=conf)
	main(sc)