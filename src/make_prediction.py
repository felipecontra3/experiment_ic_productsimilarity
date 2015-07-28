import sys, os, math, re, unicodedata
from timeit import default_timer as timer
from nltk.tag import pos_tag 
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from pymongo import MongoClient
from pyspark import SparkConf, SparkContext
from classes.Classifier import Classifier

#general variables
#MongoDB
host = '192.168.33.10'
port = 27017
username = ''
password = ''
database = 'recsysdb'

APP_NAME = 'Recomender System'
threshold  = 0.15
numMaxSuggestionsPerPost = 5

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
                   
def getTokensAndCategories():  
    db = createMongoDBConnection(host, port, username, password, database)
    model = db.model
    
    tokens_dict = db.model.find({"_type": "token"}).limit(1).next()
    del tokens_dict['_type']
    del tokens_dict['_id']
    del tokens_dict['_datetime']
    tokens_list = [None] * (max(tokens_dict.values()) + 1)

    for key, value in tokens_dict.iteritems():
        tokens_list[value] = key

    categories_dict = db.model.find({"_type": "category"}).limit(1).next()
    del categories_dict['_type']
    del categories_dict['_id']
    del categories_dict['_datetime']
    categories_list = [None] * (max(categories_dict.values()) + 1)

    for key, value in categories_dict.iteritems():
        categories_list[value] = key

    categories_and_subcategories_dict = db.model.find({"_type": "category and subcategory"}).limit(1).next()
    del categories_and_subcategories_dict['_type']
    del categories_and_subcategories_dict['_id']
    del categories_and_subcategories_dict['_datetime']
    categories_and_subcategories_list = [None] * (max(categories_and_subcategories_dict.values()) + 1)

    for key, value in categories_and_subcategories_dict.iteritems():
        pre_string = key.split(",")
        categories_and_subcategories_list[value] = (pre_string[0], pre_string[1])

    return tokens_list, categories_list, categories_and_subcategories_list

def insertSuggestions(suggestions_list, iduser, productRDD):
    
    suggestions_to_insert = []
    for post in suggestions_list:
        if len(post) > 0:
            suggestions_dict = dict()
            product_dict = dict()

            suggestions_dict['iduser'] = iduser
            suggestions_dict['idpost'] = post[0]
            suggestions_dict['post'] = post[1][0][1]
            suggestions_dict['resource'] = post[1][1]
            
            post[1][0][0].sort(key=lambda x: -x[1])
            if len(post[1][0][0]) > 0:
                suggestions_dict['suggetions'] = []
                i = 0
                for product in post[1][0][0]:
                    i = i + 1
                    product_dict = dict()
                    product_dict['produto'] = product[0]
                    product_dict['cosine_similarity'] = product[1]
                    suggestions_dict['suggetions'].append(product_dict)
                    if i == numMaxSuggestionsPerPost:
                        break

            suggestions_to_insert.append(suggestions_dict)

    db = createMongoDBConnection(host, port, username, password, database)
    db.suggestions.insert_many(suggestions_to_insert)

    return True


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

    iduser = 1
    posts = [
                (u'post1', u'I love computers! i would like to buy an asus notebook.', u'Post', u'Twitter'),
                (u'post2', u'My tablet is not working anymore, i need to buy a new one', u'Post', u'Facebook'),
                (u'post3', u'I love to watch TV on saturday nights! ', u'Post', u'Twitter'),
                (u'post4', u'i love to watch netflix on my smart tv', u'Post', u'Twitter'),
                (u'post5', u'The #Kindle2 seems the best eReader, but will it work in the UK and where can I get one?', u'Post', u'Facebook'),
                (u'post6', u'I still love my Kindle2 but reading The New York Times on it does not feel natural. I miss the Bloomingdale ads.', u'Post', u'Facebook')
            ]

    postsRDD = sc.parallelize(posts)
    tokens, category, categoryAndSubcategory = getTokensAndCategories()
    stpwrds = stopwords.words('english')
    tbl_translate = dict.fromkeys(i for i in xrange(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P') or unicodedata.category(unichr(i)).startswith('N'))

    productRDD = sc.parallelize(findProductsByCategory(category))

    productAndPostRDD = productRDD.union(postsRDD)
    
    corpusRDD = (productAndPostRDD.map(lambda s: (s[0], word_tokenize(s[1].translate(tbl_translate).lower()), s[2], s[3]))
                           .map(lambda s: (s[0], [PorterStemmer().stem(x) for x in s[1] if x not in stpwrds and x in tokens], s[2], s[3]))
                           .map(lambda s: (s[0], [x[0] for x in pos_tag(s[1]) if x[1] == 'NN' or x[1] == 'NNP'], s[2], s[3]))
                           .filter(lambda x: len(x[1]) >= 10 or x[2] == u'Post')
                           .cache())

    print corpusRDD.take(1)
    sys.exit(0)

    idfsRDD = idfs(corpusRDD)
    idfsRDDBroadcast = sc.broadcast(idfsRDD.collectAsMap())
    tfidfRDD = corpusRDD.map(lambda x: (x[0], tfidf(x[1], idfsRDDBroadcast.value), x[2], x[3])).cache()
    tfidfPostsRDD = tfidfRDD.filter(lambda x: x[2]=='Post').cache()
    tfidfPostsBroadcast = sc.broadcast(tfidfPostsRDD.map(lambda x: (x[0], x[1])).collectAsMap())

    corpusPostsNormsRDD = tfidfPostsRDD.map(lambda x: (x[0], norm(x[1]))).cache()
    corpusPostsNormsBroadcast = sc.broadcast(corpusPostsNormsRDD.collectAsMap())

    classifier = Classifier(sc, 'NaiveBayes')
    classifierDT = Classifier(sc, 'DecisionTree')

    modelNaiveBayesCategory = classifier.getModel('/dados/models/naivebayes/category')
    modelNaiveBayesSubcategory = classifier.getModel('/dados/models/naivebayes/subcategory')
    modelDecisionTree = classifierDT.getModel('/dados/models/dt/category')

    postsSpaceVectorRDD = classifier.createVectSpacePost(tfidfPostsRDD, tokens)     

    #predictionCategoryNaiveBayesCategoryRDD = postsSpaceVectorRDD.map(lambda p: modelNaiveBayesCategory.predict(p))
    #predictionCategoryDecisionTreeRDD = modelDecisionTree.predict(postsSpaceVectorRDD.map(lambda x: x))
    predictions = postsSpaceVectorRDD.map(lambda p: (modelNaiveBayesSubcategory.predict(p[1]), p[0])).groupByKey().mapValues(list).collect()     

    for prediction in predictions:

        category_to_use = categoryAndSubcategory[int(prediction[0])][0]

        tfidfProductsCategoryRDD = tfidfRDD.filter(lambda x: x[2]==category_to_use).cache()
        tfidfProductsCategoryBroadcast = sc.broadcast(tfidfProductsCategoryRDD.map(lambda x: (x[0], x[1])).collectAsMap())

        corpusInvPairsProductsRDD = tfidfProductsCategoryRDD.flatMap(lambda r: ([(x, r[0]) for x in r[1]])).cache()
        corpusInvPairsPostsRDD = tfidfPostsRDD.flatMap(lambda r: ([(x, r[0]) for x in r[1]])).filter(lambda x: x[1] in prediction[1]).cache()
        commonTokens = (corpusInvPairsProductsRDD.join(corpusInvPairsPostsRDD)
                                                 .map(lambda x: (x[1], x[0]))
                                                 .groupByKey()
                                                 .cache())
        corpusProductsNormsRDD = tfidfProductsCategoryRDD.map(lambda x: (x[0], norm(x[1]))).cache()
        corpusProductsNormsBroadcast = sc.broadcast(corpusProductsNormsRDD.collectAsMap())

        similaritiesRDD =  (commonTokens
                            .map(lambda x: cosineSimilarity(x, tfidfProductsCategoryBroadcast.value, tfidfPostsBroadcast.value, corpusProductsNormsBroadcast.value, corpusPostsNormsBroadcast.value))
                            .cache())

        suggestions = (similaritiesRDD
                        .map(lambda x: (x[0][1], (x[0][0], x[1])))
                        .filter(lambda x: x[1][1]>threshold)
                        .groupByKey()
                        .mapValues(list)
                        .join(postsRDD)
                        .join(postsRDD.map(lambda x: (x[0], x[3])))
                        .collect())

        if len(suggestions) > 0:
            insertSuggestions(suggestions, iduser, productRDD)


    elap = timer()-start
    print 'it tooks %d seconds' % elap

if __name__ == '__main__':
    conf = SparkConf().setAppName(APP_NAME)
    sc = SparkContext(conf=conf)
    main(sc)
