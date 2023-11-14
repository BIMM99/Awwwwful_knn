import pandas as pd
import math as m
import numpy as np

#ex1
#Q1
def distance(a, b):
	#1.first idea euclid = m.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)
	#2.more efficient method
	euclid = (a-b)**2
	euclid = m.sqrt(np.sum(euclid))
	return euclid

#Q2
#1.first idea, works but maybe not so efficient
def m_distance(A, b):
	l_dist =[]
	for i in range(0,A.shape[0]):
		a = np.array(A[i])
		l_dist.append(distance(a,b))
	l_dist = np.array(l_dist)
	return l_dist

#2.alternative method but results seems slighty diferent but maybe more efficient
def alt_m_dist(A, b):
	x=(A-b)**2
	x=x.sum(axis=1)
	x = np.sqrt(x)
	return x

#Q3
def k_closest(A,b,k):
	tri=np.argsort(m_distance(A, b))
	return tri[0:k]

#Q4
def mode(a):
#1.first idea --> doesn't function and would only fucntion on dataframes
	"""gr=a.groupby(level=0).count()
	max=gr[0]
	for i in range(1, len(gr)):
		if gr[i]>max :
			max=gr[i]
			mod=i
	return mod"""
#2.I saw on the net that solution : from scipy import stats and m = stats.mode(a)
#but I will try to do it myself
#3.if there is no mode I think it chooses the first occurence in the list
	values=[]
	nb=[]
	for i in range(0, len(a)):
		if a[i] not in values :
			values.append(a[i])

	for j in range(0, len(values)):
		nb.append(a.count(values[j]))
	nb = np.array(nb)
	ind_max = nb.argmax()
	mode = values[ind_max]
	return mode

#Q5
def knn(A, y, b, k):
	list_attr_closest=[]
	for i in range(0, k):
		list_attr_closest.append(y[[k_closest(A,b,k)[i]]])
	return mode(list_attr_closest)

#Q6
def mknn(A, y, B, k):
	pred=[]
	for i in range(0, len(B)):
		pred.append(knn(A, y, B[i], k)[0]) #i had to put the index 0 because it was returning me a list of arrays... I don't understand exactly why
	return pred

#Q7
def confusion(a, b):
	df = pd.DataFrame(zip(a, b),columns =['predicted', 'real value'])
	Confusion_matric = pd.crosstab(df['predicted'], df['real value'])
	return Confusion_matric

#Q8
learning_data =pd.read_csv('/Labs/Data 2d 2/2d-learn.csv')
"""X1=learning_data['X1'].to_numpy()
X2=learning_data['X2'].to_numpy()
X1X2=learning_data.drop(['Y'], axis=1).to_numpy()
Y=learning_data['Y'].to_numpy()
confusion(mknn(X1X2, Y, X1X2,1), Y)"""

#was not completing after 5 mins on my computer so trying on smaller sample (random 200 values)
learning_data=learning_data.sample(n = 200)
X1=learning_data['X1'].to_numpy()
X2=learning_data['X2'].to_numpy()
X1X2=learning_data.drop(['Y'], axis=1).to_numpy()
Y=learning_data['Y'].to_numpy()
print(confusion(mknn(X1X2, Y, X1X2,1), Y))
print(confusion(mknn(X1X2, Y, X1X2,3), Y))
print(confusion(mknn(X1X2, Y, X1X2,5), Y))


import scipy.spatial.distance as ssd
from statistics import mode
import time


def mk_closest(A,B,k):
	A2B = ssd.cdist(A, B, 'euclidean')
	C = A2B.argsort()
	return C[:,:k]

def mknn2(A, y, B, k):
	Yhat = []
	for i in range(mk_closest(A,B,k).shape[0]):
		liste = []
		for j in range(mk_closest(A,B,k).shape[1]):
			liste.append(y[mk_closest(A,B,k)[i,j]])
		Yhat.append(mode(liste))
	return np.array(Yhat)

print(confusion(mknn2(X1X2, Y, X1X2,1), Y))
print(confusion(mknn2(X1X2, Y, X1X2,3), Y))
print(confusion(mknn2(X1X2, Y, X1X2,5), Y))

before = time.time()
confusion(mknn2(X1X2, Y, X1X2,5), Y)
after = time.time()
print(after - before)

before = time.time()
confusion(mknn2(X1X2, Y, X1X2,5), Y)
after = time.time()
print(after - before)

#Ma seconde version est plus rapide


from sklearn import neighbors
from sklearn.metrics import confusion_matrix

learning_data =pd.read_csv('/Labs/Data 2d 2/2d-learn.csv')
X1X2=learning_data.drop(['Y'], axis=1).to_numpy()
Y=learning_data['Y'].to_numpy()

knearest10 = neighbors.KNeighborsClassifier(10)
knearest10.fit(X1X2, Y)
Y_hat = knearest10.predict(X1X2)


print(confusion_matrix(Y, Y_hat))

learning_data=learning_data.sample(n = 300) #I use a sample to compute quicklier
X1X2=learning_data.drop(['Y'], axis=1).to_numpy()
Y=learning_data['Y'].to_numpy()

before = time.time()
confusion(mknn2(X1X2, Y, X1X2,10), Y)
after = time.time()
print(after - before)

before = time.time()
confusion(mknn2(X1X2, Y, X1X2,10), Y)
after = time.time()
print(after - before)

before = time.time()
confusion_matrix(Y, knearest10.predict(X1X2))
after = time.time()
print(after - before)

#clear difference in favor of the last implementation, the 2 other ones cant compute the full set

learning_data =pd.read_csv('/Labs/Data 2d 2/2d-learn.csv')
eval_data = pd.read_csv('/Labs/Data 2d 2/2d-eval.csv')
eval_lbl = pd.read_csv('/Labs/Data 2d 2/2d-eval-labels.csv')
obsY = eval_lbl['Y']
optY = eval_lbl['Yopt']
print(confusion_matrix(obsY, knearest10.predict(eval_data)))
print(confusion_matrix(optY, knearest10.predict(eval_data)))
#the optimal model should be with a different value of k


# on crée une boucle qui nous trouve le k qui minimise le risque (fonction valant 1 pour chaque erreur)
# jusqu'à un k max sur la base eval --> obtenir le modèle optimal
# on pose un k max sinon il faudrait le faire pour tout k
def optKnn(X,Y,X_test, Y_test, k_max):
	risk=5000
	min=1
	for i in range(2,k_max-1):
		knearest = neighbors.KNeighborsClassifier(i)
		knearest.fit(X, Y)
		if confusion_matrix(Y_test, knearest.predict(X_test))[1][0] + confusion_matrix(Y_test, knearest10.predict(X_test))[0][1]<risk:
			min=i
			risk= confusion_matrix(Y_test, knearest.predict(X_test))[1][0] + confusion_matrix(Y_test, knearest10.predict(X_test))[0][1]
	return min

print(optKnn(X1X2, Y, X1X2, Y, 40)) #modèle optimal sur les données d'apprentissages  en évaluant le risque sur ces mêmes données
# k =3
print(optKnn(X1X2, Y, eval_data, obsY, 40)) # j'ai essayé de trouver le modèle basé sur les données d'apprentissage
# minimisant le risque sur les données d'évaluation --> l'algorithme est trop lent

knearest3 = neighbors.KNeighborsClassifier(3)
knearest3.fit(X1X2, Y)
print(confusion_matrix(optY, knearest3.predict(eval_data)))

knearest5 = neighbors.KNeighborsClassifier(5)
knearest5.fit(X1X2, Y)
print(confusion_matrix(optY, knearest5.predict(eval_data)))

knearest7 = neighbors.KNeighborsClassifier(7)
knearest7.fit(X1X2, Y)
print(confusion_matrix(optY, knearest7.predict(eval_data)))

knearest15 = neighbors.KNeighborsClassifier(15)
knearest15.fit(X1X2, Y)
print(confusion_matrix(optY, knearest15.predict(eval_data)))

#autre méthode pour retrouver le modèle optimal
for i in range(1,41):
	knearest = neighbors.KNeighborsClassifier(i)
	knearest.fit(X1X2, Y)
	if confusion_matrix(optY, knearest.predict(eval_data))[1][0] + confusion_matrix(optY, knearest10.predict(eval_data))[0][1]==0:
		print(i)
		break
