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
                (u'post498', u'Reading the tweets coming out of Iran... The whole thing is terrifying and incredibly sad...', u'post', u'post'),
                (u'post497', u'Trouble in Iran, I see. Hmm. Iran. Iran so far away. #flockofseagullsweregeopoliticallycorrect', u'post', u'post'),
                (u'post496', u'Ahhh... back in a *real* text editing environment. I &lt;3 LaTeX.', u'post', u'post'),
                (u'post495', u'On that note, I hate Word. I hate Pages. I hate LaTeX. There, I said it. I hate LaTeX. All you TEXN3RDS can come kill me now.', u'post', u'post'),
                (u'post494', u'Ask Programming: LaTeX or InDesign?: submitted by calcio1 [link] [1 comment] http://tinyurl.com/myfmf7', u'post', u'post'),
                (u'post493', u'After using LaTeX a lot, any other typeset mathematics just looks hideous.', u'post', u'post'),
                (u'post492', u'using Linux and loving it - so much nicer than windows... Looking forward to using the wysiwyg latex editor!', u'post', u'post'),
                (u'post491', u'I just created my first LaTeX file from scratch. That didnt work out very well. (See @amandabittner , its a great time waster)', u'post', u'post'),
                (u'post490', u'i lam so in love with Bobby Flay... he is my favorite. RT @terrysimpson: @bflay you need a place in Phoenix. We have great peppers here!', u'post', u'post'),
                (u'post489', u'@johncmayer is Bobby Flay joining you?', u'post', u'post'),
                (u'post488', u'getting ready to test out some burger receipes this weekend. Bobby Flay has some great receipes to try. Thanks Bobby.', u'post', u'post'),
                (u'post487', u'Twitter Stock buzz: $AAPL $ES_F $SPY $SPX $PALM  (updated: 12:00 PM)', u'post', u'post'),
                (u'post486', u'Monday already. Iran may implode. Kitchen is a disaster. @annagoss seems happy. @sebulous had a nice weekend and @goldpanda is great. whoop.', u'post', u'post'),
                (u'post485', u'Shits hitting the fan in Iran...craziness indeed #iranelection', u'post', u'post'),
                (u'post484', u'How to Track Iran with Social Media: http://bit.ly/2BoqU', u'post', u'post'),
                (u'post483', u'7 hours. 7 hours of inkscape crashing, normally solid as a rock. 7 hours of LaTeX complaining at the slightest thing. I cant take any more.', u'post', u'post'),
                (u'post482', u'@Iheartseverus we love you too and dont want you to die!!!!!!  Latex = the devil', u'post', u'post'),
                (u'post481', u'Fighting with LaTex. Again...', u'post', u'post'),
                (u'post480', u'My dad was in NY for a day, we ate at MESA grill last night and met Bobby Flay. So much fun, except I completely lost my voice today.', u'post', u'post'),
                (u'post479', u'cant wait for the great american food and music festival at shoreline tomorrow.  mmm...katz pastrami and bobby flay. yes please.', u'post', u'post'),
                (u'post478', u'Gonna go see Bobby Flay 2moro at Shoreline. Eat and drink. Gonna be good.', u'post', u'post'),
                (u'post477', u'Excited about seeing Bobby Flay and Guy Fieri tomorrow at the Great American Food &amp; Music Fest!', u'post', u'post'),
                (u'post476', u'has a date with bobby flay and gut fieri from food network', u'post', u'post'),
                (u'post475', u'dearest @google, you rich bastards! the VISA card you sent me doesnt work. why screw a little guy like me?', u'post', u'post'),
                (u'post474', u'Off to the bank to get my new visa platinum card', u'post', u'post'),
                (u'post473', u'@ruby_gem My primary debit card is Visa Electron.', u'post', u'post'),
                (u'post472', u'I have a google addiction. Thank you for pointing that out, @annamartin123. Hahaha.', u'post', u'post'),
                (u'post471', u'The #Kindle2 seems the best eReader, but will it work in the UK and where can I get one?', u'post', u'post'),
                (u'post470', u'@cwong08 I have a Kindle2 (&amp; Sony PRS-500). Like it! Physical device feels good. Font is nice. Pg turns are snappy enuf. UI a little klunky.', u'post', u'post'),
                (u'post469', u'Man I kinda dislike Apple right now. Case in point: the iPhone 3GS. Wish there was a video recorder app. Please?? http://bit.ly/DZm1T', u'post', u'post'),
                (u'post468', u'@sklososky Thanks so much!!! ...from one of your *very* happy Kindle2 winners ; ) I was so surprised, fabulous. Thank you! Best, Kathleen', u'post', u'post'),
                (u'post467', u'Missed this insight-filled May column: One smart guy looking closely at why hes impressed with Kindle2 http://bit.ly/i0peY @wroush', u'post', u'post'),(u'post466', u'I hope the girl at work  buys my Kindle2', u'post', u'post'),
                (u'post465', u'looks like summize has gone down. too many tweets from WWDC perhaps?', u'post', u'post'),
                (u'post464', u'GOT MY WAVE SANDBOX INVITE! Extra excited! Too bad I have class now... but Ill play with it soon enough! #io2009 #wave', u'post', u'post'),
                (u'post463', u'Today is a good day to dislike AT&amp;T. Vote out of office indeed, @danielpunkass', u'post', u'post'),
                (u'post462', u'Fuzzball is more fun than AT&amp;T ;P http://fuzz-ball.com/twitter', u'post', u'post'),
                (u'post461', u'@Plip Where did you read about tethering support Phil?  Just AT&amp;T or will O2 be joining in?', u'post', u'post'),
                (u'post460', u'@freitasm oh I see. I thought AT&amp;T were 900MHz WCDMA?', u'post', u'post'),
                (u'post459', u'@sheridanmarfil - its not so much my obsession with cell phones, but the iphone!  im a slave to at&amp;t forever because of it. :)', u'post', u'post'),
                (u'post458', u'Although todays keynote rocked, for every great announcement, AT&amp;T shit on us just a little bit more.', u'post', u'post'),(u'post457', u'I love my Kindle2. No more stacks of books to trip over on the way to the loo.', u'post', u'post'),
                (u'post456', u'I still love my Kindle2 but reading The New York Times on it does not feel natural. I miss the Bloomingdale ads.', u'post', u'post'),
                (u'post455', u'Id say some sports writers are idiots for saying Roger Federer is one of the best ever in Tennis.  Roger Federer is THE best ever in Tennis', u'post', u'post'),
                (u'post454', u'I cant watch TV without a Tivo.  And after all these years, the Time/Warner DVR  STILL sucks. http://www.davehitt.com/march03/twdvr.html', u'post', u'post'),
                (u'post453', u'I mean, Im down with Notre Dame if I have to.  Its a good school, Id be closer to Dan, Id enjoy it.', u'post', u'post'),
                (u'post452', u'reading Michael Palin book, The Python Years...great book. I also recommend Warren Buffet &amp; Nelson Mandelas bio', u'post', u'post'),
                (u'post451', u'Im truly braindead.  I couldnt come up with Warren Buffets name to save my soul', u'post', u'post'),
                (u'post450', u'SUPER INVESTORS: A great weekend read here from Warren Buffet. Oldie, but a goodie. http://tinyurl.com/oqxgga', u'post', u'post')
            ]

    postsRDD = sc.parallelize(posts)
    tokens, category, categoryAndSubcategory = getTokensAndCategories()
    categs = ["Computers & Tablets", "Video Games", "TV & Home Theater"]# , "Musical Instruments"]
    stpwrds = stopwords.words('english')
    tbl_translate = dict.fromkeys(i for i in xrange(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P') or unicodedata.category(unichr(i)).startswith('N'))

    productRDD = sc.parallelize(findProductsByCategory(categs))
    #productAndPostRDD = productRDD.union(postsRDD)
    productAndPostRDD = sc.parallelize(productRDD.collect()+ postsRDD.collect())
    
    corpusRDD = (productAndPostRDD.map(lambda s: (s[0], word_tokenize(s[1].translate(tbl_translate).lower()), s[2], s[3]))
                           .map(lambda s: (s[0], [PorterStemmer().stem(x) for x in s[1] if x not in stpwrds], s[2], s[3] ))
                           .map(lambda s: (s[0], [x for x in s[1] if x in tokens], s[2], s[3] ))
                           .cache())

    idfsRDD = idfs(corpusRDD)
    idfsRDDBroadcast = sc.broadcast(idfsRDD.collectAsMap())
    tfidfRDD = corpusRDD.map(lambda x: (x[0], tfidf(x[1], idfsRDDBroadcast.value), x[2], x[3]))
    tfidfPostsRDD = tfidfRDD.filter(lambda x: x[0]=='post452')    

    classifier = Classifier(sc, 'NaiveBayes')
    classifierDT = Classifier(sc, 'DecisionTree')

    modelNaiveBayesCategory = classifier.getModel('/dados/models/naivebayes/category')
    modelNaiveBayesSubcategory = classifier.getModel('/dados/models/naivebayes/subcategory')
    modelDecisionTree = classifierDT.getModel('/dados/models/dt/category')

    postsSpaceVectorRDD = classifier.createVectSpacePost(tfidfPostsRDD, tokens)

    predictionCategoryNaiveBayesCategoryRDD = postsSpaceVectorRDD.map(lambda p: modelNaiveBayesCategory.predict(p))
    predictionCategoryNaiveBayesSubcategoryRDD = postsSpaceVectorRDD.map(lambda p: modelNaiveBayesSubcategory.predict(p))
    predictionCategoryDecisionTreeRDD = modelDecisionTree.predict(postsSpaceVectorRDD.map(lambda x: x))
    
    print 'NB Category %d' % predictionCategoryNaiveBayesCategoryRDD.take(1)[0]
    print 'NB Subategory %d' % predictionCategoryNaiveBayesSubcategoryRDD.take(1)[0]
    print 'DT Category %d' % predictionCategoryDecisionTreeRDD.take(1)[0]

    elap = timer()-start
    print 'it tooks %d seconds' % elap

if __name__ == '__main__':
    conf = SparkConf().setAppName(APP_NAME)
    sc = SparkContext(conf=conf)
    main(sc)