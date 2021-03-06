import sys, os, math, datetime
from timeit import default_timer as timer
import re, unicodedata
from nltk.tag import pos_tag 
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from pymongo import MongoClient
from pyspark import SparkConf, SparkContext
from classes.Classifier import Classifier

host = '192.168.33.10'
port = 27017
username = ''
password = ''
database = 'recsysdb'

APP_NAME = 'Recomender System - Spark Job'

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

def insertTokensAndCategories(tokens, category, categoryAndSubcategory):
    db = createMongoDBConnection(host, port, username, password, database)

    modelCollection = db.model
    modelCollection.remove({'_type':'token'})

    document_mongo =  dict()
    document_mongo['_type'] = 'token'
    document_mongo['_datetime'] = datetime.datetime.utcnow()
    i = 0
    for t in tokens:
        document_mongo[t] = i
        i = i + 1   

    modelCollection.insert_one(document_mongo)

    modelCollection.remove({'_type':'category'})

    document_mongo =  dict()
    document_mongo['_type'] = 'category'
    document_mongo['_datetime'] = datetime.datetime.utcnow()
    i = 0
    for c in category:
        document_mongo[c] = i
        i = i + 1 

    modelCollection.insert_one(document_mongo)

    modelCollection.remove({'_type':'category and subcategory'})
    
    document_mongo =  dict()
    document_mongo['_type'] = 'category and subcategory'
    document_mongo['_datetime'] = datetime.datetime.utcnow()
    i = 0
    for c in categoryAndSubcategory:
        document_mongo[c[0]+","+c[1]] = i
        i = i + 1 

    modelCollection.insert_one(document_mongo)

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
    import math
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

def main(sc):
    start = timer()

    categs = ["Computers & Tablets", "Video Games", "TV & Home Theater"]# , "Musical Instruments"]

    stpwrds = stopwords.words('english')
    tbl_translate = dict.fromkeys(i for i in xrange(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('S') or unicodedata.category(unichr(i)).startswith('P') or unicodedata.category(unichr(i)).startswith('N'))

    productRDD = sc.parallelize(findProductsByCategory(categs))
    corpusRDD = (productRDD.map(lambda s: (s[0], word_tokenize(s[1].translate(tbl_translate).lower()), s[2], s[3]))
						   .map(lambda s: (s[0], [PorterStemmer().stem(x) for x in s[1] if x not in stpwrds], s[2], s[3] ))
                           .map(lambda s: (s[0], [x[0] for x in pos_tag(s[1]) if x[1] == 'NN' or x[1] == 'NNP'], s[2], s[3]))
                           .cache())

    idfsRDD = idfs(corpusRDD)
    idfsRDDBroadcast = sc.broadcast(idfsRDD.collectAsMap())
    tfidfRDD = corpusRDD.map(lambda x: (x[0], tfidf(x[1], idfsRDDBroadcast.value), x[2], x[3]))

    category = productRDD.map(lambda x: x[2]).distinct().collect()
    categoryAndSubcategory = productRDD.map(lambda x: (x[2], x[3])).distinct().collect()
    tokens = corpusRDD.flatMap(lambda x: x[1]).distinct().collect()


    insertTokensAndCategories(tokens, category, categoryAndSubcategory)
    
    classifier = Classifier(sc, 'NaiveBayes')
    trainingVectSpaceCategoryRDD, testVectSpaceCategoryRDD = classifier.createVectSpaceCategory(tfidfRDD, category, tokens).randomSplit([8, 2], seed=0L)
    modelNaiveBayesCategory = classifier.trainModel(trainingVectSpaceCategoryRDD, '/dados/models/naivebayes/category_new')

    predictionAndLabelCategoryRDD = testVectSpaceCategoryRDD.map(lambda p : (category[int(modelNaiveBayesCategory.predict(p.features))], category[int(p.label)]))
    acuraccyCategory = float(predictionAndLabelCategoryRDD.filter(lambda (x, v): x[0] == v[0]).count())/float(predictionAndLabelCategoryRDD.count())
    print 'the accuracy of the Category Naive Bayes model is %f' % acuraccyCategory

    #training in this second way just for test
    
    trainingVectSpaceSubcategory, testVectSpaceSubcategory = classifier.createVectSpaceSubcategory(tfidfRDD, categoryAndSubcategory, tokens).randomSplit([8, 2], seed=0L)
    modelNaiveBayesSubcategory = classifier.trainModel(trainingVectSpaceSubcategory, '/dados/models/naivebayes/subcategory_new')

    predictionAndLabelSubcategory = testVectSpaceSubcategory.map(lambda p : (categoryAndSubcategory[int(modelNaiveBayesSubcategory.predict(p.features))], categoryAndSubcategory[int(p.label)]))
    acuraccySubcategory = float(predictionAndLabelSubcategory.filter(lambda (x, v): x[0] == v[0]).count())/float(predictionAndLabelSubcategory.count())
    print 'the accuracy of the Subcategory Naive Bayes model is %f' % acuraccySubcategory

    #test with DecisionTree Model
    classifierDT = Classifier(sc, 'DecisionTree')
    trainingVectSpaceCategory, testVectSpaceCategory = classifierDT.createVectSpaceCategory(tfidfRDD, category, tokens).randomSplit([8, 2], seed=0L)
    modelDecisionTreeCategory = classifierDT.trainModel(trainingVectSpaceCategory, '/dados/models/dt/category_new')

    predictions = modelDecisionTreeCategory.predict(testVectSpaceCategory.map(lambda x: x.features))
    predictionAndLabelCategory = testVectSpaceCategory.map(lambda lp: lp.label).zip(predictions)
    acuraccyDecisionTree = float(predictionAndLabelCategory.filter(lambda (x, v): x == v).count())/float(predictionAndLabelCategory.count())   
    print 'the accuracy of the Decision Tree model is %f' % acuraccyDecisionTree

    elap = timer()-start
    print 'it tooks %d seconds' % elap

if __name__ == '__main__':
    conf = SparkConf().setAppName(APP_NAME)
    sc = SparkContext(conf=conf)
    main(sc)