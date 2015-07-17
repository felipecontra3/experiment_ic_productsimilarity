import sys, os, shutil
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector

class NaiveBayesClassifier:

	def __init__(self, sc):
		self.sc = sc

	def createVectSpaceCategory(self, featureCorpus, category, tokens):
		numTokens = len(tokens)
		return featureCorpus.map(lambda t: LabeledPoint(category.index(t[2]) , SparseVector(numTokens, sorted([tokens.index(i) for i in t[1].keys()]), [t[1][tokens[i]] for i in sorted([tokens.index(i) for i in t[1].keys()])])))

	def createVectSpaceSubcategory(self, featureCorpus, category, tokens):
		numTokens = len(tokens)
		return featureCorpus.map(lambda t: LabeledPoint(category.index((t[2], t[3])) , SparseVector(numTokens, sorted([tokens.index(i) for i in t[1].keys()]), [t[1][tokens[i]] for i in sorted([tokens.index(i) for i in t[1].keys()])])))		

	''' I need to improve the creation. What happens if a model alread exists in path?'''
	def trainModel(self, vectSpace, path):
		try:
			model = NaiveBayes.train(vectSpace)
			model.save(self.sc, path)
		except:
			print "Unexpected error:", sys.exc_info()[0]
		 	raise
		return model

	def getModel(self, path):
		return NaiveBayesModel.load(self.sc, path)


	''' I NEED TO CORRECT THIS, it is not working'''
	def deleteModel(self, path):
		for the_file in os.listdir(path):
			file_path = os.path.join(path, the_file)
			try:
				if os.path.isfile(file_path):
					os.unlink(file_path)
				#elif os.path.isdir(file_path): shutil.rmtree(file_path)
			except Exception, e:
				print e

	def testModel(self, featureCorpus):
		return True

	



