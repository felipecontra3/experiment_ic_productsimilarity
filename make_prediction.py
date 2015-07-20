import sys, os, math
from timeit import default_timer as timer
import re, unicodedata
import nltk
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from pymongo import MongoClient
from pyspark import SparkConf, SparkContext
from Classifier import Classifier

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
		
		category = None; fatherCategory = None
		if 'categorias' in keys:
			Subcategory = produto['categorias'][2]['nome']; Category = produto['categorias'][1]['nome']

		product_list.append((id, description, Category, Subcategory))

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
    start = timer()

    posts = [
                (u'post1', u"I love computers! i would like to buy an asus notebook.", u'Post', u'Post'),
                (u'post2', u"My tablet is not working anymore, i need to buy a new one", u'Post', u'Post'),
                (u'post3', u"I love to watch TV on saturday nights!", u'Post', u'Post'),
                (u'post4', u"i love to watch netflix on my smart tv", u'Post', u'Post')
            ]


    postRDD = sc.parallelize(posts)

    categs = ["Computers & Tablets", "Video Games", "TV & Home Theater"]# , "Musical Instruments"]

    stpwrds = stopwords.words('english')
    tbl_translate = dict.fromkeys(i for i in xrange(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P') or unicodedata.category(unichr(i)).startswith('N'))

    productRDD = sc.parallelize(findProductsByCategory(categs))

    productAndPostRDD = productRDD.union(postRDD)
    corpusRDD = (productAndPostRDD.map(lambda s: (s[0], word_tokenize(s[1].translate(tbl_translate).lower()), s[2], s[3]))
                           .map(lambda s: (s[0], [PorterStemmer().stem(x) for x in s[1] if x not in stpwrds], s[2], s[3] )))

    corpusRDD_ = (productRDD.map(lambda s: (s[0], word_tokenize(s[1].translate(tbl_translate).lower()), s[2], s[3]))
                           .map(lambda s: (s[0], [PorterStemmer().stem(x) for x in s[1] if x not in stpwrds], s[2], s[3] )))

    idfsRDD = idfs(corpusRDD)
    idfsRDDBroadcast = sc.broadcast(idfsRDD.collectAsMap())
    tfidfRDD = corpusRDD.map(lambda x: (x[0], tfidf(x[1], idfsRDDBroadcast.value), x[2], x[3]))
    tfidfPostsRDD = tfidfRDD.filter(lambda x: x[0]=='post4')
    
    tokens = corpusRDD_.flatMap(lambda x: x[1]).distinct().collect()

    classifier = Classifier(sc, 'NaiveBayes')
    classifierDT = Classifier(sc, 'DecisionTree')

    modelNaiveBayesCategory = classifier.getModel('/dados/models/naivebayes/category')
    modelNaiveBayesSubcategory = classifier.getModel('/dados/models/naivebayes/subcategory')
    modelDecisionTree = classifierDT.getModel('/dados/models/dt/category')

    postsSpaceVectorRDD = classifier.createVectSpacePost(tfidfPostsRDD, tokens)

    predictionCategoryNaiveBayesCategoryRDD = postsSpaceVectorRDD.map(lambda p: modelNaiveBayesCategory.predict(p))
    predictionCategoryNaiveBayesSubcategoryRDD = postsSpaceVectorRDD.map(lambda p: modelNaiveBayesSubcategory.predict(p))
    predictions = modelDecisionTree.predict(postsSpaceVectorRDD.map(lambda x: x))
    
    category = productRDD.map(lambda x: x[2]).distinct().collect()
    categoryAndSubcategory = productRDD.map(lambda x: (x[2], x[3])).distinct().collect()

    print 'NB Category %d' % predictionCategoryNaiveBayesCategoryRDD.take(1)[0]
    print 'NB Subategory %d' % predictionCategoryNaiveBayesSubcategoryRDD.take(1)[0]
    print 'DT Category %d' % predictions.take(1)[0]

    elap = timer()-start
    print 'it tooks %d seconds' % elap

if __name__ == '__main__':
    conf = SparkConf().setAppName(APP_NAME)
    sc = SparkContext(conf=conf)
    main(sc)