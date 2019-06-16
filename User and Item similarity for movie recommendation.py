import numpy as np
import math
from numpy.linalg import inv
from numpy import sqrt
import pandas as pd
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from numpy import ravel

# Load data into a dataframe and convert into a matrix of float64 datatype (for faster computations)
shows = pd.read_csv('shows.txt', sep=" ", header=None)
shows=np.asmatrix(shows)

# Create a dictionary for showid and showname
showDict = {}
for key in range(shows.size):
	showDict[key] = shows[key,0]
	
	
# Load data into a dataframe and convert into a matrix of float64 datatype (for faster computations)
data = pd.read_csv('user-shows.txt', sep=" ", header=None)
data=np.asmatrix(data)
R=data.astype(np.float64)

# Fetch 100 movies data for user 20 into an array
user20_data = np.array(R[19,:100])

# Movies watched by user20 (element value is 1)
user20_movies = np.where(user20_data==1)[1]

print("************************")
print("Movies watched by User20")
print("************************")
print(user20_movies)
for key in range(user20_movies.size):
	print(showDict[user20_movies[key]])

#Mark the first hundred entries of user20 as zeroes
R[19,:100]=0

#Find transpose of the user*item matrix
RT = np.transpose(R)

# Find production of the 2 matrices 
# This will yield user*user matrix 
P = np.matmul(R,RT)
Q = np.matmul(RT,R)
RRT = P
RTR = Q

###################################
# Calculate User Similarity matrix#
###################################

# Mark all non-diagonal elements as zero
P=np.diag(np.diag(P))
P_inv=inv(sqrt(P))

# Generate User similarity matrix
SU=np.matmul((np.matmul((np.matmul(P_inv,R)),RT)),P_inv)

## Generate User-User Recommendation matrix
TU=np.matmul(SU,R)

## Get maximum value in user20
maxVal = TU[19,:].max()
userSim_Normal=TU[19,:]/maxVal

# Movies to be recommended based on 70% acceptance
userSim_reco = np.where(userSim_Normal>0.7)[1]

# Filter movies in 1-100 category
userSim_reco = userSim_reco[userSim_reco<100]

# Show 1 to 100 Recommendations for User 20
print("*******************************************************************")
print("Movies that can be recommended for User20 by User-User Similarity method")
print("*******************************************************************")
for key in userSim_reco:
	print(showDict[key])

###################################
# Calculate Item Similarity matrix#
###################################

# Mark all non-diagonal elements as zero
Q=np.diag(np.diag(Q))
Q_inv=inv(sqrt(Q))

# Generate User similarity matrix
SI=np.matmul((np.matmul((np.matmul(Q_inv,RT)),R)),Q_inv)

## Generate User-User Recommendation matrix
TI=np.matmul(R,SI)

## Get maximum value in user20
maxVal = TI[19,:].max()
itemSim_Normal=TI[19,:]/maxVal

# Movies to be recommended based on 70% acceptance
itemSim_reco = np.where(itemSim_Normal>0.7)[1]

# Filter movies in 1-100 category
itemSim_reco = itemSim_reco[itemSim_reco<100]

# Show 1 to 100 Recommendations for User 20
print("*******************************************************************")
print("Movies that can be recommended for User20 by Item-Item Similarity method")
print("*******************************************************************")
for key in itemSim_reco:
	print(showDict[key])

##################################################
# Function to get the top5 scores and its indices
# If there is a match, pick the lowest index
##################################################
def printTop5(inputData,inputDataIndex,max,func):
	k=0
	i=99
	outputData = PrettyTable(['Movie Name', 'Score'])
	outList = []
	exitLoop=0
	score_before=0
	while(exitLoop==0):
		score=round(inputData[0,i],4)
		if(score!=score_before):
			movieName=showDict[inputDataIndex[0,i]]
			outputData.add_row([movieName,score])
			outList.append(inputDataIndex[0,i])
			k=k+1
		i=i-1
		score_before=score
		if(k==max):
			exitLoop=1
	if(func=='print'):
		print(outputData)
	else:
		return(np.asarray(outList))

##################################################
# Function to get the true positive rate from k 
# values being 1 to 19
##################################################
def getTruePositiveRate(vector,vector_index):
	k=1
	k_list = []
	trueRate_list = []
	while(k<20):
		vec=printTop5(vector,vector_index,k,'list')
		matched=np.intersect1d(vec,user20_movies)
		trueRate=(matched.size)/totalwatched
		k_list.append(k)
		trueRate_list.append(trueRate)
		k=k+1
	return(k_list,trueRate_list)

	
## Sort user recommendation matrix
## Display top 5 similar scores
user_sorted=np.sort(TU[19,:100])
user_sorted_ix=np.argsort(TU[19,:100])

print("************************************************")
print("Top 5 shows from User-User recommendation matrix")
print("************************************************")
printTop5(user_sorted,user_sorted_ix,5,'print')

## Sort Item recommendation matrix
## Display top 5 similar scores
item_sorted=np.sort(TI[19,:100])
item_sorted_ix=np.argsort(TI[19,:100])

print("************************************************")
print("Top 5 shows from Item-Item recommendation matrix")
print("************************************************")
printTop5(item_sorted,item_sorted_ix,5,'print')

t=printTop5(user_sorted,user_sorted_ix,5,'list')
totalwatched=user20_movies.size

(xVal,yVal) = getTruePositiveRate(user_sorted,user_sorted_ix)
plt.plot(xVal,yVal,'b-',label='User-User Similarity')
(xVal,yVal) = getTruePositiveRate(item_sorted,item_sorted_ix)
plt.plot(xVal,yVal,'r-',label='Item-Item Similarity')
plt.legend()
plt.ylabel('True Positive Rate')
plt.xlabel('k-value')
plt.xticks(np.arange(min(xVal), max(xVal)+1, 1.0))
plt.show()