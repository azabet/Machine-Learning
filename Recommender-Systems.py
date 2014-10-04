
# coding: utf-8

## Recommender Systems:

## Content-Based and Collaborative Filtering

# ___Goal:___ Provide movie recommendations based on users' prior ratings.
# 
# ___Data:___ 100,000 movie ratings from 1000 users on 1700 movies, downloaded from [GroupLens.org](http://grouplens.org/datasets/movielens/).
# 
# ___Methods:___
# 1. <u>_Content-Based Filtering_</u>: Model the ratings based on user information (age, gender) and movie genres (action, comedy, etc.) using _Support Vector Machines_ (SVM) regression.  This method requires prior knowledge of movie features (genres).
# 2. <u>_k-Nearest Neightbors_</u> (kNN):  Calculate the similarity between users based on their ratings on common movies.  Then, take the average rating of nearest-neightbors to predict the ratings of unrated movies.  This method does not require knowledge of movie features.
# 3. <u>_Collaborative Filtering_</u>: Extract both the movie features and user preferences directly from the data by concurrently optimizating the parameters through _linear regression_. 

## Initialize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cross_validation import KFold


## Explore the data
data = pd.read_table('u.data').drop('timestamp', axis=1)
N = len(data)
print data.shape
print data.head(2)


## Number of users:
Nu = len(data.userID.unique())
print Nu


## Number of movies:
Nm = len(data.movieID.unique())
print Nm


## Rating distribution:
data.rating.value_counts(sort=False).plot(kind='bar')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


## Number of movies per users:
movies_per_users = data.userID.value_counts()
print 'min = %d, mean = %d, max = %d' %(movies_per_users.min(), 
                                        movies_per_users.mean(),
                                        movies_per_users.max())


## Number of users per movies:
users_per_movies = data.movieID.value_counts()
print 'min = %d, mean = %d, max = %d' %(users_per_movies.min(), 
                                        users_per_movies.mean(),
                                        users_per_movies.max())


## Convert data to matrix
rating = data.pivot(index='userID', columns='movieID').rating
userID = rating.index
movieID = rating.columns
print rating.shape
print rating.head(2)


## 1. Content-Based Filtering

## Load user information
users_info = pd.read_table('u.user', sep='|').set_index('userID')
print users_info.shape
print users_info.head(2)


## Create age groups
## The age bins are adapted from the IMDb website.
users_info['age_group'] = pd.cut(users_info.age, [0,18,29,45,np.inf])
users_info.drop(['age', 'occupation', 'zipcode'], axis=1, inplace=True)
print users_info.head(2)


## Load movie information
movies_info = pd.read_table('u.item', sep='|').set_index('movieID')#.drop(low_count_movies)
movies_info['release date'] = pd.to_datetime(movies_info['release date'])
print movies_info.shape
print movies_info.head(2)


## Join data together
joined = data.join(users_info, on='userID').join(movies_info.iloc[:,4:], on='movieID')
joined.sort(['userID', 'movieID'], inplace=True)
print joined.shape
print joined.head(2)


## Calculate average ratings by gender and age group
## The average rating will be used as an additional movie parameter.
avg = joined.groupby(['movieID', 'gender', 'age_group']).rating.mean()
newdata = joined.join(avg, rsuffix='_avg', on=['movieID', 'gender', 'age_group'])
newdata.drop(['gender', 'age_group'], axis=1, inplace=True)
newdata.set_index(['userID', 'movieID'], inplace=True)
print newdata.head(2)


## Support Vector Machines (SVM)
## Apply SVM to model the ratings of each user based on the movie parameters.
def SVM(Xtrain, ytrain, Xtest=None, C=1):
    model = SVR(C=C)  ## module imported from Scikit-Learn
    model.fit(Xtrain, ytrain)
    pred = model.predict(Xtrain)
    if Xtest is None:
        return pred
    else:
        pred_test = model.predict(Xtest)
    return pred, pred_test

X = newdata.drop('rating', axis=1)
Y = newdata.rating


## Evaluation metric: root-mean-squared error (RMSE)
def rmse(y, yhat):
    e2 = (y - yhat) ** 2
    return np.sqrt(e2.mean())


## k-Fold Cross-Validation
## Optimize the regularization parameter (C) by cross-validation.  The purpose of regularization is to minimize overfitting/underfitting (bias/variance trade off).
cost = [0.1, 0.3, 1, 3, 10, 30]
error, CVerror = [], []
for C in cost:
    for user in userID:
        x, y = X.ix[user], Y.ix[user]
        err, CVerr = [], [] 
        for train, test in KFold(len(y), n_folds=3, shuffle=True):
            Xtrain, Xtest = x.ix[x.index[train]], x.ix[x.index[test]]
            ytrain, ytest = y.ix[x.index[train]], y.ix[x.index[test]]
            pred_train, pred_test = SVM(Xtrain, ytrain, Xtest, C=C)
            ## training error
            err.append(rmse(ytrain, pred_train))
            ## cross-validation error
            CVerr.append(rmse(ytest, pred_test))
    err_avg, CVerr_avg = sum(err)/len(err), sum(CVerr)/len(CVerr)
    error.append(err_avg)
    CVerror.append(CVerr_avg)
    print C, '\t', err_avg, '\t', CVerr_avg


## Plot the training and cross-validation errors
ax = plt.gca()
plt.plot(cost, error, '-o')
plt.plot(cost, CVerror, '-o', color='r')
ax.set_xscale('log')
plt.legend(['training', 'cross-validation'], loc=3)
plt.xlabel('Cost')
plt.ylabel('Error (RMSE)')
plt.title('SVM Optimization')
plt.show()


## C = 1 gives the lowest cross-validation error.


## Plot the predictions
def jitter(y, yhat, title=''):
    pred = np.round(yhat)
    pred[pred > 5] = 5
    pred[pred < 1] = 1
    def noise():
        return np.random.randn(len(y)) * 0.1
    plt.scatter(y + noise(), pred + noise())
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title(title)
    plt.axis('image')
    plt.grid()
    plt.show()
    print 'error (rmse) =', rmse(y, yhat)
    
user = 1
x, y = X.ix[user], Y.ix[user]
yhat = SVM(x, y, C=1)
jitter(y, yhat, 'Content-Based Filtering')


## 2. k-Nearest Neighbors (kNN)

## Calculate pairwise distances between users
def dist(i,j):
    ## Only consider users who have at least 3 movies in common
    d = (rating.ix[i] - rating.ix[j])**2
    if d.count() >= 3:
        return np.sqrt(d.mean())

## Distance matrix
D = np.empty((Nu,Nu))
D.fill(np.nan)
D = pd.DataFrame(D)
D.index = D.columns = userID
for user1 in userID:
    for user2 in userID[userID > user1]:
        D[user1].ix[user2] = dist(user1, user2)
        D[user2].ix[user1] = D[user1].ix[user2]


## Predict ratings by the average of nearest neighbors
def kNN_predict(user, k=5):
    ## Sort users by distance
    neighbors = D[user].order().index

    ## Function for calculting the kNN average
    def kNN_average(x):
        return x.ix[neighbors].dropna().head(k).mean()

    ## Apply kNN_average to every movie
    pred = rating.apply(kNN_average, axis=0)
    return pred


## Optimize k
user = 1
K = [3, 5, 10, 15]
error = []
for k in K:
    y = rating.ix[user]
    yhat = kNN_predict(user, k=k)
    err = rmse(y, yhat)
    error.append(err)
    print k, '\t', err

plt.plot(K, error, '-o')
plt.xlabel('# Nearest Neighbors')
plt.ylabel('Error (RMSE)')
plt.show()


## k = 5 gives the lowest error.


## Plot the predictions
user = 1
pred = kNN_predict(user, k=5)
jitter(rating.ix[user], pred, 'kNN')


## 3. Collaborative Filtering

## In this method, we apply linear regression to fit both the user parameters, _U_, and the movie features, _M_.  We first initialize them to small random values, then fit _U_ and _M_ recursively while keeping the other constant.  In order to have enough data points for the linear regression, we only include movies that have at least 3 ratings.
low_count_movies = users_per_movies[users_per_movies < 3].index
newdata = data.set_index('movieID').drop(low_count_movies).reset_index()
print newdata.shape
print N = len(newdata)

rating = newdata.pivot(index='userID', columns='movieID').rating
userID, movieID = rating.index, rating.columns
Nu, Nm = len(userID), len(movieID)
print Nu, Nm


## Initialize the parameters

## number of movie features
n = 20

## Matrix of user preferences
U = pd.DataFrame(np.random.rand(Nu, n), index=userID, columns=range(1, n+1))

## Matrix of movie features
M = pd.DataFrame(np.random.rand(Nm, n), index=movieID, columns=range(1, n+1))


## Fit user preferences
def fitU():  
    print 'Fitting user parameters',
    
    ## Join ratings and movie features
    Udata = newdata.set_index('movieID').join(M).sort('userID').set_index('userID')
    
    ## Function for fitting individual users
    model = LinearRegression(fit_intercept=False)
    def Ufit(i):
        if i % 100 == 0:
            print '.',
        df = Udata.ix[i]
        X = df.drop('rating', axis=1)
        y = df.rating
        model.fit(X, y)
        return model.coef_

    ## Fit all users
    for i in userID:
        U.ix[i] = Ufit(i)
    
    ## Calculate the error
    pred, error = predict(U, M)
    print '\n  error (rmse) =', error,
    error_history.append(error)
    delta = bestFit['error'] - error
    if delta > 0:
        print ', improved by', delta
        bestFit['U'] = U
        bestFit['M'] = M
        bestFit['error'] = error
    else:
        print ', increased by', -delta
    return delta

def predict(U, M):
    pred = U.dot(M.T)
    pred[pred > 5] = 5
    pred[pred < 1] = 1
    e2 = (pred - rating) ** 2
    error = np.sqrt(e2.sum().sum() / N)
    return pred, error


## Fit movie features
def fitM():
    print 'Fitting movie features',
    
    ## Join ratings and user preferences
    Mdata = newdata.set_index('userID').join(U).sort('movieID').set_index('movieID')
    
    ## Function for fitting individual movies
    model = LinearRegression(fit_intercept=False)
    def Mfit(j):
        if j % 100 == 0:
            print '.',
        df = Mdata.ix[j]
        X = df.drop('rating', axis=1)
        y = df.rating
        model.fit(X, y)
        return model.coef_

    ## Fit all movies 
    for j in movieID:
        M.ix[j] = Mfit(j)
    
    ## Calculate the error
    pred, error = predict(U, M)
    print '\n  error (rmse) =', error,
    error_history.append(error)
    delta = bestFit['error'] - error
    if delta > 0:
        print ', improved by', delta
        bestFit['U'] = U
        bestFit['M'] = M
        bestFit['error'] = error
    else:
        print ', increased by', -delta
    return delta


## Fit both U and M
error_history = []
bestFit = {'U': U, 'M': M, 'error': np.inf}
delta = fitU()
tolerance = 0.0001
while delta > tolerance:
    delta = fitM()
    if delta > tolerance:
        delta = fitU()
        
plt.plot(error_history, '-o')
plt.xlabel('Iteration')
plt.ylabel('Error (RMSE)')
plt.show()


## Plot the predictions
user = 1
jitter(rating.ix[user], U.ix[user].dot(M.T), 'Collaborative Filtering')


## Movie recommendations
def recommend(user):
    unrated_movies = rating.ix[user][rating.ix[user].isnull()].index
    pred, error = predict(U.ix[user], M.ix[unrated_movies])
    movies = movies_info.ix[unrated_movies]
    movies['predicted rating'] = pred
    top_movies = movies.sort(['predicted rating', 'release date'], 
                                  ascending=False, inplace=False).head()
    return top_movies[['title','predicted rating']]

recommend(1)


## Conclusions

# * Collaborative filtering using linear regression gives the lowest error in predicting the movie ratings, since it can optimize both the user preferences and the movie features.
# * The model does not require prior knowledge of movie features, unlike content-based filtering, therefore can have broader applications.
# * The number of movie features can be optimized through cross-validation.
