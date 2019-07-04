
import os
#os.chdir('Dropbox/code/money_and_happiness')

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# analyze r29 money diary posts: how does article sentiment covary with author 
# income, author demographics, author occupation, etc.?

figpath = 'figures/'

dictionary = pd.read_csv('data/inquirercleaned.csv')
article_data = pd.read_csv('data/article_data.csv')
article_data[['location','industry','occupation','period']] = article_data[['location','industry','occupation','period']].fillna('')

# generate the (# articles) x (# topics) article-topic matrix: 
# i.e. entry (a,k) is number of words from topic k in article a
for i in list(range(len(article_data))):
    content = article_data.loc[i,'content']
    word_ct_vec = np.asmatrix(dictionary.Entry.apply(lambda w: len(re.findall(w,content))))
    try:
        article_word_matrix = np.concatenate((article_word_matrix,word_ct_vec),axis=0)
    except NameError:
        article_word_matrix = word_ct_vec

word_topic_matrix = np.asmatrix(dictionary.drop(columns=['Entry']))
article_topic_matrix = np.matmul(article_word_matrix,word_topic_matrix)

# standardize article-topic matrix to perform pca
colmeans = article_topic_matrix.mean(0)
colstds = article_topic_matrix.std(0)
article_topic_matrix_std = np.divide(article_topic_matrix-np.repeat(colmeans,np.size(article_topic_matrix,0),axis=0),np.repeat(colstds,np.size(article_topic_matrix,0),axis=0))
        
# potential outcome variables of interest (i.e. measures of sentiment based on 
# article text): 

# i - first PC of article topic matrix (does this weight particularly strongly
# on any topic(s)?)
U, s, W = np.linalg.svd(article_topic_matrix_std,full_matrices=False)

pc1_wts = pd.Series(W.T[:,0].flatten().tolist()[0],index=dictionary.drop(columns=['Entry']).columns)
# interpretation: the first pc weights approximately equally on all topics,
# except for minimal weight on the "feel" topic.
# this makes interpretation of the first pc fairly challenging; leave this be 
# for now.

# ii - (+) article score = number of "positive" words in article, scaled by the
# total number of words in the article
article_word_count_vec = np.asmatrix(article_data['content'].str.split(' ').apply(len)).T

idx_pos = [i == 'Pstv' for i in dictionary.drop(columns=['Entry']).columns]
pos_score = np.divide(article_topic_matrix[:,idx_pos],article_word_count_vec)

# visualize the distribution of this sentiment measure:
plt.hist(pos_score)

# iii - (-) article score = number of "negative" words in article
idx_neg = [i == 'Ngtv' for i in dictionary.drop(columns=['Entry']).columns]
neg_score = np.divide(article_topic_matrix[:,idx_neg],article_word_count_vec)

# visualize the distribution of this sentiment measure:
plt.hist(neg_score)

# what's the correlation between positive and negative article scores?
np.corrcoef(np.concatenate((pos_score,neg_score),axis=1).T)
# the correlation is low, positive; i.e. articles with more positive words also
# tend, weakly, to include more negative words (in a linear form).

# iv - "net" positive score
# (trying to isolate meaningful variation on the (+)/(-) scale given that the
# levels of these two emotions seem to move together somewhat)
net_pos_score = np.divide(article_topic_matrix[:,idx_pos]-article_topic_matrix[:,idx_neg],
                          article_word_count_vec)

# visualize the distribution of this sentiment measure:
plt.hist(net_pos_score)

# exploratory analysis:

# scatter plots of sentiment against yearly income
yearly_income = list(article_data['yearly_amt'].values)

fig, ax = plt.subplots(figsize=(10,5))  
plt.scatter(yearly_income,net_pos_score.T.tolist()[0])
plt.xlabel('Yearly income ($)')
plt.ylabel('Net positive score')
plt.savefig(figpath + 'scatter_all')

# interpretation:
# there's no obvious relationship between income and net (+) score: this could
# be + or -, but looks roughly flat in a best-fit sense.
# it is interesting to note the conical shape of the scatter: i.e. the variance 
# of net positive sentiment appears to be decreasing in income. 
# this suggests that at lower levels of income, different individuals can have 
# vastly different levels of relative happiness. (likely determined, to some
# extent, by observable and unobservable factors.) however, at higher levels of
# income, different individuals' levels of relative happiness tend to be more 
# similar, and closer to the average value of net positivity (~= 0, or an exact
# balance of positive and negative expression).

# to look for a clearer relationship = between income and sentiment, consider 
# this conditional on other covariates that we observe:

# industry
industry_group = article_data.groupby('industry').industry.count().sort_values(ascending=False)
top_industries = industry_group[:7].index

r = 0
c = 0
fig, ax = plt.subplots(4,2,sharey='all',figsize=(6,15))
plt.subplots_adjust(hspace = 0.7)
for i in top_industries:
    idx_industry = article_data['industry'] == i
    article_sub = list(article_data.loc[idx_industry,'yearly_amt'].values)
    net_pos_sub = net_pos_score[idx_industry].T.tolist()[0]
    ax[r,c].scatter([j/1000 for j in article_sub],net_pos_sub)
    ax[r,c].set_xlabel('Yr. income ($1000)')
    ax[r,c].set_ylabel('Net pos. score')
    ax[r,c].set_title(i)
    if r == 3:
        r = 0
        c = 1
    else:
        r += 1
idx_other = ~article_data['industry'].isin(top_industries)
article_sub = list(article_data.loc[idx_other,'yearly_amt'].values)
net_pos_sub = net_pos_score[idx_other].T.tolist()[0]
ax[3,1].scatter([j/1000 for j in article_sub],net_pos_sub)
ax[3,1].set_xlabel('Yr. income ($1000)')
ax[3,1].set_ylabel('Net pos. score')
ax[3,1].set_title('Other')
plt.savefig(figpath + 'scatter_industry')

# interpretation: 
# for the most part, there is still not a clear relationship btwn. income level
# and sentiment conditional on industry.
# there is arguably a negative linear relationship btwn. these variables in 
# tech, law, and government.
# there is arguably a positive linear relationship btwn. these variables in 
# education, although this may not hold at higher income levels.

# income type: joint vs individual
joint_types = {'Individual':False,'Joint':True}

c = 0
fig, ax = plt.subplots(1,2,sharey='all',figsize=(6,3))
plt.subplots_adjust(hspace = 0.7)
for key, value in joint_types.items():
    idx_joint = article_data['is_joint'] == value
    article_sub = list(article_data.loc[idx_joint,'yearly_amt'].values)
    net_pos_sub = net_pos_score[idx_joint].T.tolist()[0]
    ax[c].scatter([j/1000 for j in article_sub],net_pos_sub)
    ax[c].set_xlabel('Yr. income ($1000)')
    ax[c].set_ylabel('Net pos. score')
    ax[c].set_title(key)
    c += 1
plt.savefig(figpath + 'scatter_joint')    
    
# interpretation:
# there may be a slight positive linear relationship btwn. income level and 
# sentiment conditional on a joint income.
    
# income type: hourly/monthly vs. salaried
period_types = {'Hourly':['Hour','Month'],'Salaried':['']}

c = 0
fig, ax = plt.subplots(1,2,sharey='all',figsize=(6,3))
plt.subplots_adjust(hspace = 0.7)
for key, value in period_types.items():
    idx_period = article_data['period'].isin(value)
    article_sub = list(article_data.loc[idx_period,'yearly_amt'].values)
    net_pos_sub = net_pos_score[idx_period].T.tolist()[0]
    ax[c].scatter([j/1000 for j in article_sub],net_pos_sub)
    ax[c].set_xlabel('Yr. income ($1000)')
    ax[c].set_ylabel('Net pos. score')
    ax[c].set_title(key)
    c += 1
plt.savefig(figpath + 'scatter_hourly')

# interpretation:
# there appears to be a slight negative linear relationship btwn. income level 
# and sentiment, conditional on an hourly income.
    
# age
age_brackets = {'20-29 yrs.':list(range(20,30)),
                '30-39 yrs.':list(range(30,40)),
                '40+ yrs.': list(range(40,int(max(article_data['age']))+1))}

c = 0
fig, ax = plt.subplots(1,3,sharey='all',figsize=(9,3))
plt.subplots_adjust(hspace = 0.7)
for key, value in age_brackets.items():
    idx_age = article_data['age'].isin(value)
    article_sub = list(article_data.loc[idx_age,'yearly_amt'].values)
    net_pos_sub = net_pos_score[idx_age].T.tolist()[0]
    ax[c].scatter([j/1000 for j in article_sub],net_pos_sub)
    ax[c].set_xlabel('Yr. income ($1000)')
    ax[c].set_ylabel('Net pos. score')
    ax[c].set_title(key)
    c += 1
plt.savefig(figpath + 'scatter_age')
    
# interpretation:
# no clear relationship between income and sentiment conditional on age.

# in general, there does not appear to be a strong association between 
# income and sentiment, even when conditioning on other covariates (i.e. 
# including one level of interaction).
# as an alternative, consider a nonlinear predictor of sentiment conditional on
# income and the other individual-level covariates.
# a random forest regression would allow for greater non-linearities and help 
# capture higher-order interactions between these features. additionally, we 
# could get some insight into which of these features are most important in  
# predicting sentiment score by looking at variable importance measures. (e.g.
# is income level even a top predictor of sentiment?)

from sklearn.ensemble import RandomForestRegressor

# prepare data for use in random forest regression
article_data_rf = article_data[['location','is_joint','is_periodic','occupation','industry','age','yearly_amt']]
article_data_rf['net_pos_score'] = net_pos_score

# there aren't many missing values, so use a "cheap" imputation method
for numeric_fd in ['age','yearly_amt']:
    idx_nan = np.isnan(article_data_rf[numeric_fd])
    impute = article_data_rf.loc[~idx_nan,numeric_fd].mode().values[0]
    article_data_rf.loc[idx_nan,numeric_fd] = impute

# convert categorical features into ordinal features by ordering the categories
# by increasing mean of the outcome variable (sentiment). 
# (see Hastie, Tibshirani, and Friedman, "The Elements of Statistical Learning"
# ch. 9 p. 310)
make_ordinal = ['location','occupation','industry']
for categorical_fd in make_ordinal:
    idx_missing = article_data_rf[categorical_fd].apply(len) == 0
    article_data_rf.loc[idx_missing,categorical_fd] = 'missing_' + categorical_fd
    category_ordering = article_data_rf.groupby(categorical_fd).net_pos_score.mean().sort_values()
    category_ordinal = pd.DataFrame(category_ordering.index).reset_index().rename(columns={'index':categorical_fd + '_ordinal'})
    article_data_rf = article_data_rf.merge(category_ordinal,how='inner',on=categorical_fd)

article_data_rf = article_data_rf.drop(columns=make_ordinal)

# feature and outcome matrices
X = article_data_rf[['is_joint','is_periodic','age','yearly_amt','location_ordinal','occupation_ordinal','industry_ordinal']].values
y = article_data_rf['net_pos_score'].values

# subset training data, testing data     
idx_train = np.random.choice(list(range(np.size(X,0))),300,replace=False)

# there are three tuning parameters for us to consider:
# m = number of candidate features to draw for each node split,
# n = minimum leaf node size,
# B = number of trees
# since the oob error rate of a given rf model approximates the test error, we
# can use this as the criterion that we want to maximize for tuning. this will
# save time versus 10-fold cv.
# (see T. Hastie, R. Tibshirani and J. Friedman, “Elements of Statistical 
# Learning Ed. 2”, p. 592-593, Springer, 2009)
m_list = list(range(1,np.size(X,1)+1))
n_list = list(range(5,151,5))
B_list = list(range(1,501,50))
oob_errors = np.empty((len(m_list),len(n_list),len(B_list)))   

for i in list(range(len(m_list))):
    m = m_list[i]
    for j in list(range(len(n_list))):
        n = n_list[j]
        for k in list(range(len(B_list))):
            B = B_list[k]
            warm = (B>1)
            rf = RandomForestRegressor(n_estimators=B, 
                                       criterion='mse',
                                       min_samples_leaf=n,
                                       max_features=m,
                                       bootstrap=True, 
                                       oob_score=True, 
                                       warm_start=warm)
            rf.fit(X[idx_train,:],y[idx_train])
            oob_err = 1 - rf.oob_score_
            oob_errors[i,j,k] = oob_err
            if B == 500:
                print('finished estimating forest for (m,n) = ',m,n)

tuning_df = pd.DataFrame({'m':[0]*len(B_list),'n':[0]*len(B_list),'B':[0]*len(B_list),'oob_err':[0]*len(B_list)})
for i in list(range(len(B_list))):
    slc = oob_errors[:,:,i]
    idx_min = np.unravel_index(np.argmin(slc, axis=None), slc.shape)
    tuning_df.loc[i,'m'] = m_list[idx_min[0]]
    tuning_df.loc[i,'n'] = n_list[idx_min[1]]
    tuning_df.loc[i,'B'] = B_list[i]
    tuning_df.loc[i,'oob_err'] = slc[idx_min[0],idx_min[1]]
   
# note: oob error (1 - r2) reaches global minimum of ~23.5% with m = 5, n = 5,
# B = 450.   

# visualize change in oob error rate (~= test error rate) as forest size grows
# also visualize optimal m,n conditional on forest size
fig, ax1 = plt.subplots()

ax1.set_xlabel('Number of trees')
ax1.set_ylabel('OOB error rate')
l1 = ax1.plot(tuning_df['B'],tuning_df['oob_err'],color='C0',label='OOB error rate')

ax2 = ax1.twinx()
ax2.set_ylabel('m/n')
l2 = ax2.plot(tuning_df['B'],tuning_df['m'],color='C1',label='No. candidate features per node split')
l3 = ax2.plot(tuning_df['B'],tuning_df['n'],color='C2',label='Min. leaf node size')

ax1.legend(l1+l2+l3,[l.get_label() for l in l1+l2+l3])

fig.tight_layout()
plt.savefig(figpath + 'tuning')

# interpretation: we see that oob error rate levels off at around 100 trees 
# when selecting optimal m, n conditional on B. 
# since the global min oob error is attained with m = 5, n = 5, B = 450 but we 
# see little improvement in oob error with 300+ trees, fit a final model with 
# m = 5, n = 5, B = 250.

B = 250
n = 5
m = 5        

rf = RandomForestRegressor(n_estimators=B, 
                           criterion='mse',
                           min_samples_leaf=n,
                           max_features=m,
                           bootstrap=True, 
                           oob_score=True, 
                           warm_start=False)
# fit chosen model to training set, obtain test error in test set (rmse, r2)
rf.fit(X[idx_train,:],y[idx_train])
idx_test = np.setdiff1d(list(range(np.size(X,0))),idx_train)
yhat = rf.predict(X[idx_test,:])
rmse_test = np.sqrt(np.mean(np.square(yhat-y[idx_test]))) 
# test rmse ~= 0.0056. this is a bit under 50% of 1 sd of y.
r2_test = rf.score(X[idx_test,:],y[idx_test]) 
# test r2 ~= 82.6%  

# which variables are most "important" in improving random forest predictions?
# i.e. which variables produce the largest (average) decrease in node mse when
# splitting on that variable across all trees in the forest? # (see T. Hastie, 
# R. Tibshirani and J. Friedman, “Elements of Statistical Learning Ed. 2”, p. 
# 368, Springer, 2009)
# (note that scikit learn reports these measures scaled relative to each other
# st. they sum to one)
split_vars= ['is_joint','is_periodic','age','yearly_amt','location_ordinal','occupation_ordinal','industry_ordinal']
fig, ax = plt.subplots()
ax.bar(split_vars,rf.feature_importances_)
ax.set_xticklabels(split_vars,rotation=45 )
plt.savefig(figpath + 'variable_importance')

# interpretation: 
# variables related to one's job are very important based on the "importance" 
# criteria: occupation accounts for over 60% of the decrease in net positive 
# score fitting error. occupation and industry jointly account for almost 80% 
# of the error decrease.
# the amount of one's salary is relatively unimportant, accounting for only 1%
# of the decrease in net positive score fitting error. this is unsurprising 
# given the above results, which indicated that income level didn't have a 
# strong association with net positive score, even when conditioning on other
# covariates. 
