
# coding: utf-8

## Clustering Analysis of Dopant Distribution

# ___Goal:___ Analysis of dopant clustering in graphene, a novel 2D material. <br>
# ___Data:___ Dopant coordinates (in nanometers) and sublattice positions (_A_ or _B_) -- data from original research. <br>
# ___Methods:___ Classification using <u>_Logistic Regression_</u> and <u>_Support Vector Machines_</u>, and unsupervised clustering using a personally developed model implementing <u>_Spatial Autocorrelation Analysis_</u>. <br>
# ___Outcome:___ Discovered large-scale segregation of dopant domains in graphene which led to the discovery of a new mechanism of doping.  The autocorrelation model was able to detect an unknown number of clusters, unlike logistic regression or SVM in which the number of clusters should be pre-specified.  The model also achieved detection of nonlinear domain boundaries.  Results are published in [JACS 136, 1391 (2014)](http://dx.doi.org/10.1021/ja408463g).  

### 0. Background

# The purpose of this study is to analyze the patterns of dopant distribution in graphene, resulting from various doping methods, in order to understand and better control the doping process.  Two doping methods were developed: (1) chemical vapor deposition (_Sample 1_), and (2) reaction with ammonia (_Sample 2_).  
# 
# Data consists of dopant coodinates in nanometers, and sublattice positions with regards to graphene's unit cell (_A_ or _B_).  The goal is to detect the clustering of dopants with respect to the sublattice position.  Note that in pristine graphene the sublattice positions are equivalent, therefore they are expected to be uniformly occupied by the dopants.

### 1. Data Visualization

# In[1]:

# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Set the maximum number of rows for displaying tables
pd.options.display.max_rows = 5

sample1 = pd.read_csv('sample1.csv')
sample2 = pd.read_csv('sample2.csv')
sample1


# In[2]:

fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_figwidth(10)

# Plot Sample1
gp = sample1.groupby('Sublattice')
gp.get_group('A').plot(x='X', y='Y', style='b^', ax=ax1)
gp.get_group('B').plot(x='X', y='Y', style='rv', ax=ax1)
ax1.set_title('Sample 1')
ax1.set_xlabel('X'); ax1.set_ylabel('Y')
ax1.set_xlim(0,100); ax1.set_ylim(100,0)
ax1.axis('image')

# Plot Sample2
gp = sample2.groupby('Sublattice')
gp.get_group('A').plot(x='X', y='Y', style='b^', ax=ax2)
gp.get_group('B').plot(x='X', y='Y', style='rv', ax=ax2)
ax2.set_title('Sample 2')
ax2.set_xlabel('X'); ax2.set_ylabel('Y')
ax2.set_xlim(0,50); ax2.set_ylim(50,0)
ax2.axis('image')
plt.legend(['A','B'], bbox_to_anchor=(1.35,0.65), title='Sublattice')
plt.show()


# In Sample 1, the red and blue dopants seem to be well segregated, whereas in Sample 2 the dopant distribution appears to be more disordered.  The objective is to identify the sizes of the clustering domains as well as the domain boundaries.

### 2. Feature Normalization

# Rescale the coordinates to the [0,1] range, and create binary values for the sublattices:

# In[3]:

def normalize(data):
    data['x'] = (data.X - data.X.min()) / (data.X.max() - data.X.min())
    data['y'] = (data.Y - data.Y.min()) / (data.Y.max() - data.Y.min())
    data['Type'] = 0
    data['Type'][data.Sublattice == 'B'] = 1

normalize(sample1)
normalize(sample2)
sample1


### 3. Logistic Regression

# In[4]:

def LR(data, cost=1.0):
    X = np.matrix([data.x, data.y]).T
    y = data.Type
    model = LogisticRegression(fit_intercept=True, C=cost)
    model.fit(X, y)
    visualize(model)
    plotData(data)

def visualize(model):
    gridX = np.arange(0, 1, 0.01)
    gridY = np.arange(0, 1, 0.01)
    xx, yy = np.meshgrid(gridX, gridY)
    grid = pd.DataFrame({'x' : xx.ravel(), 'y' : yy.ravel()})
    X = np.matrix([grid.x, grid.y]).T
    yhat = model.predict(X)
    yhat = np.int_(yhat.reshape(xx.shape))
    ax.imshow(yhat, origin='lower', cmap='bwr', extent=[0,1,0,1])

def plotData(data):
    gp = data.groupby('Type')
    gp.get_group(0).plot(x='x', y='y', style='k^', ylim=(1,0), ax=ax)
    gp.get_group(1).plot(x='x', y='y', style='wv', ylim=(1,0), ax=ax)
    ax.axis('image')
    ax.set_xlabel('')

fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_figwidth(10)
fig.suptitle('Logistic Regression', y=1.05, fontsize=20)
ax=ax1; LR(sample1); ax.set_title('Sample 1')
ax=ax2; LR(sample2); ax.set_title('Sample 2')
plt.show()


# For Sample 1, logistic regression can identify a linear boundary between the red and blue domains since only two clusters are present.  But for Sample 2 it fails to correctly identify the clusters.

### 4. Support Vector Machines

# In[5]:

def SVM(data, cost=1.0):
    X = np.matrix([data.x, data.y]).T
    y = data.Type
    model = SVC(C=cost)
    model.fit(X, y)
    visualize(model)
    plotData(data)

fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_figwidth(10)
fig.suptitle('Support Vector Machines', y=1.05, fontsize=20)
ax=ax1; SVM(sample1); ax.set_title('Sample 1')
ax=ax2; SVM(sample2); ax.set_title('Sample 2')
plt.show()


# SVM provides slight improvement to logistic regression -- it can detect a slighltly nonlinear domain boundary in Sample 1 -- but for Sample 2 it too performs poorly.

### 5. Unsupervised Clustering: Spatial Autocorrelation

# We have developed a nonlinear clustering model utilizing autocorrelation analysis, in which the local densities of dopants in either sublattice are calculated at every point of the map.  From the ratios of the local densities, the degree of segregation (purity of cluster) is computed, which in turn is used to evaluate the cluster sizes.

# In[6]:

def AutoCorr(data, alpha=2):
    # Create a grid map
    gridRange = np.arange(0, 1, 0.02)
    xgrid, ygrid = np.meshgrid(gridRange, gridRange)

    # Calculate the distances of dopants to the grid points
    Xgrid, Xdata = np.meshgrid(xgrid.ravel(), data.x)
    Ygrid, Ydata = np.meshgrid(ygrid.ravel(), data.y)
    R = sqrt((Xgrid - Xdata)**2 + (Ygrid - Ydata)**2)
    
    # Use the average nearest-neighbor separation (Ravg) as an
    # internal measure for calculating cluster size:
    # N * pi * Ravg**2 = area = 1.0  =>  Ravg = sqrt(1.0 / N / pi)
    Ravg = sqrt(1.0 / len(data) / pi)

    # Calculate the local densities using exponential downweighting
    # so to give higher weights to closer neighbors
    weight = np.exp(-R/alpha/Ravg)  ## alpha is the exp. decay constant
    type0 = np.nonzero(data.Type == 0)[0]
    type1 = np.nonzero(data.Type == 1)[0]
    N0 = sum(weight[type0,:], axis=0)  ## weighted sum of type 0 dopants
    N1 = sum(weight[type1,:], axis=0)  ## weighted sum of type 1 dopants
    
    # Calculate the density ratio and the degree of segregation
    ratio = N1 / (N0 + N1)
    def segregation(x, pos=None):  ## 'pos' is used later for the colorbar
        if x < 0.5:
            return (1 - x) * 100
        else:
            return x * 100
    seg = np.array(map(segregation, ratio))
    print 'Average Segregation = %.2f %% \t\t' %seg.mean(),

    # Visualize
    ratio = ratio.reshape(xgrid.shape)
    seg = seg.reshape(xgrid.shape)
    img = ax.imshow(ratio, extent=[0,1,1,0], cmap='bwr')
    img.set_clim(0,1)
    plt.colorbar(img, ax=ax, label='Segregation (%)', 
                 format=FuncFormatter(segregation))
    plotData(data)

fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_figwidth(10)
fig.suptitle('Spatial Autorcorrelation', y=1.05, fontsize=20)
ax=ax1; AutoCorr(sample1); ax.set_title('Sample 1')
ax=ax2; AutoCorr(sample2); ax.set_title('Sample 2')
plt.show()


# The colormap shows the segregation of dopants at every point of the map.  The segregation ranges from 50% (even distribution) to 100% (complete segregation).  Dark red and blue colors indicate clusters where either sublattice is dominant.  White indicates domain boundary.  

### Parameter Optimization: Overfitting vs. Underfitting

# The only parameter for the model is  `alpha`  which is a decay constant for the exponential downweighting.  Larger  `alpha`  corresponds to less downweighting, which results in including a larger radius when calculating the local densities.

###### alpha = 1

# In[7]:

fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_figwidth(10)
ax=ax1; AutoCorr(sample1, 1); ax.set_title('Sample 1')
ax=ax2; AutoCorr(sample2, 1); ax.set_title('Sample 2')
plt.show()


###### alpha = 3

# In[8]:

fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_figwidth(10)
ax=ax1; AutoCorr(sample1, 3); ax.set_title('Sample 1')
ax=ax2; AutoCorr(sample2, 3); ax.set_title('Sample 2')
plt.show()


# - Small `alpha` could lead to overfitting.
# - Large `alpha` could lead to underfitting.

### Conclusion

# Our analysis shows that samples prepared by method 1 exhibit a large degree of dopant clustering (~ 85 - 90% segregation), whereas samples produced by method 2 are essentially disordered (~ 60% segregation) and cluster sizes are much smaller.
