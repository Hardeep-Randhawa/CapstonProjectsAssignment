#Word2Vec is one of the most popular pre trained word embeddings developed by Google.
#It is trained on Good news dataset which is an extensive dataset. As the name suggests,
#it represents each word with a collection of integers known as a vector.
#The vectors are calculated such that they show the semantic relation between words."

!pip install --upgrade numpy gensim scipy pandas


#import gensim library
#from gensim.models import Word2Vec
from gensim.models import KeyedVectors


#replace with the path where you have downloaded your model.
pretrained_model_path = '/content/sample_data/GoogleNews-vectors-negative300.bin.gz'


word_vectors = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True)

# Calculate cosine similarity between word pairs
word1 = "early"
word2 = "seats"
#calculate the similarity
similarity1 = word_vectors.similarity(word1, word2)
#print final value
print(similarity1)

word3 = "king"
word4 = "man"
#calculate the similarity
similarity2 = word_vectors.similarity(word3, word4)
#print final value
print(similarity2)