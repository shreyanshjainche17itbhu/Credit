import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('cr_loan2.csv')
original = pd.read_csv('cr_loan2.csv')

print(train.dtypes)
train.head()
train.isnull().sum()
train.columns
#12 columns are present

plt.hist(train['loan_int_rate'], bins='auto', alpha=0.7)
plt.xlabel('Loan interest rate')
plt.show()

plt.hist(train['loan_amnt'], bins='auto', alpha=0.7)
plt.xlabel('Loan Amount')
plt.show()
#10000 having highest frequency

plt.scatter(train['person_income'], train['person_age'], alpha=0.2)
plt.xlabel('Person income')
plt.ylabel('person age')
plt.show()

#crosstable similar to pivoting
pd.crosstab(train['loan_intent'], train['loan_status'], margins=True)
pd.crosstab(train['person_home_ownership'], train['loan_grade'],train['loan_status'], margins=True, aggfunc='mean')
train.boxplot(column=['loan_percent_income'], by=['loan_status'])

#Detection of Outliers
print(pd.crosstab(train['loan_status'],train['person_home_ownership'],values=train['person_emp_length'], aggfunc='max'))
pd.crosstab(train['loan_grade'], train['loan_status'], values=train['loan_int_rate'], aggfunc='mean', margins=True)
sns.boxplot(train['loan_status'], train['person_emp_length'])
pd.crosstab(train['loan_status'],train['person_home_ownership'],values=train['person_age'], aggfunc=['min','max'])
plt.scatter(train['person_emp_length'], train['loan_int_rate'])

#Removal of outliers
indices = train[train['person_emp_length'] > 60].index
train = train.drop(indices)
print(pd.crosstab(train['loan_status'],train['person_home_ownership'],values=train['person_emp_length'], aggfunc=['min','max']))
indices = train[train['person_age'] > 100].index
train = train.drop(indices)

print(train.columns[train.isnull().any()])
print(train[train['person_emp_length'].isnull()].head())
train['person_emp_length'].fillna((train['person_emp_length'].median()), inplace=True)
n, bins, patches = plt.hist(train['person_emp_length'], bins='auto', color='blue')
plt.xlabel("Person Employment Length")
plt.show()

#filling of null values
train.isnull().sum()
plt.hist(train['person_emp_length'], bins='auto')
plt.scatter(train['person_emp_length'], train['person_age'])
np.median(train['person_emp_length'])
train['person_emp_length'].fillna(train['person_emp_length'].median(), inplace=True)

#for filling of loan interest rate we will take according to grade of loan 
pd.crosstab(train['loan_grade'], train['loan_status'], values=train['loan_int_rate'], aggfunc='mean', margins=True)
for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    print(i)
    train['loan_int_rate'][train['loan_grade'] == i] = train['loan_int_rate'][train['loan_grade'] == i].fillna(train['loan_int_rate'][train['loan_grade'] == i].mean())
    
train['loan_int_rate'][train['loan_grade'] == 'A'] = train['loan_int_rate'][train['loan_grade'] == 'A'].fillna(train['loan_int_rate'][train['loan_grade'] == 'A'].mean())
train['loan_int_rate'][train['loan_grade'] == 'B'] = train['loan_int_rate'][train['loan_grade'] == 'B'].fillna(train['loan_int_rate'][train['loan_grade'] == 'B'].mean())
train['loan_int_rate'][train['loan_grade'] == 'C'] = train['loan_int_rate'][train['loan_grade'] == 'C'].fillna(train['loan_int_rate'][train['loan_grade'] == 'C'].mean())
train['loan_int_rate'][train['loan_grade'] == 'D'] = train['loan_int_rate'][train['loan_grade'] == 'D'].fillna(train['loan_int_rate'][train['loan_grade'] == 'D'].mean())
train['loan_int_rate'][train['loan_grade'] == 'E'] = train['loan_int_rate'][train['loan_grade'] == 'E'].fillna(train['loan_int_rate'][train['loan_grade'] == 'E'].mean())
train['loan_int_rate'][train['loan_grade'] == 'F'] = train['loan_int_rate'][train['loan_grade'] == 'F'].fillna(train['loan_int_rate'][train['loan_grade'] == 'F'].mean())
train['loan_int_rate'][train['loan_grade'] == 'G'] = train['loan_int_rate'][train['loan_grade'] == 'G'].fillna(train['loan_int_rate'][train['loan_grade'] == 'G'].mean())

train.isnull().sum()


sns.heatmap(train, annot=True)
calc_data = plot_data.train[np.isnan(plot_data.train)]







#Applying logistic regression

#Probability of default
#The liikelihood that someone will default on loan is the probability of default
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

#applying one hot encoding to convert caategorical column
train_cat = train.select_dtypes(include=['object'])
train_num = train.select_dtypes(exclude=['object'])


train_cat = pd.get_dummies(train_cat)
train = pd.concat([train_num, train_cat], axis=1)
train.dtypes

X = train.drop('loan_status', axis=1)
y = train['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, stratify= y)

model = LogisticRegression(solver='lbfgs')

model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_test)

preds_df = pd.DataFrame(pred_proba[:, 1][:5], columns = ['prob_default'])
actual = y_test.head(5)
print(pd.concat([actual.reset_index(drop=True), preds_df], axis=1))

preds_df = pd.DataFrame(pred_proba[:, 1], columns = ['prob_default'])
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)
preds_df['loan_status'].value_counts()

model.score(X_test, y_test)
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))
#Recall = true positive predicte out of actual positives present
#Precision = true positive predicted out of all positive predicted

#Calculating Roc-Auc Score
fallout, sesitivity, thresholds = roc_curve(y_test, preds_df['prob_default'])
plt.plot(fallout, sesitivity)
plt.plot([0,1], [0,1], linestyle = '--')
auc = roc_auc_score(y_test, preds_df['prob_default'])
plt.show()

#using Confuion matrix to check number of false negative(predicted = non default, actual = default)
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)
print(confusion_matrix(y_test, preds_df['loan_status']))

preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)
print(confusion_matrix(y_test, preds_df['loan_status']))

#Calculating the loss due to prediction of false negative
default = preds_df['loan_status'].value_counts()[1]
default_recall = precision_recall_fscore_support(y_test,preds_df['loan_status'])[1][1]
avg_amnt = np.mean(train['loan_amnt'])
print(avg_amnt*(1 -default_recall)*default)

#Now we have to check the trade off between threshold value and recall 
thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
default_recalls = []
non_default_recalls = []
for i in thresh:
    preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > i else 0)
    default_recall = precision_recall_fscore_support(y_test,preds_df['loan_status'])[1][1]
    default_recalls.append(default_recall)
    non_default_recall = precision_recall_fscore_support(y_test,preds_df['loan_status'])[0][1] 
    non_default_recalls.append(non_default_recall)

plt.plot(thresh, default_recalls)
plt.plot(thresh, non_default_recalls)
plt.legend(["default_recall", "non_default_recall"])
plt.show()
# It conclude that threshold value should be between 0.3 to 0.4 for best model

#Now applying XGBoost model in place of gradien boosting model
pip install xgboost
import xgboost as xgb
gbt = xgb.XGBClassifier(learning_rate =0.1, max_depth=7)
gbt.fit(X_train, y_train)

xg_preds_proba = gbt.predict_proba(X_test)[:,1]
xg_preds = gbt.predict(X_test)

target_name = ['Non-Default', 'Default']
print(classification_report(y_test, xg_preds, target_names = target_name))

#getting the importance of different columns
gbt.get_booster.get_score(importance_type = 'weight')
#or

xgb.plot_importance(gbt, importance_type = 'weight')
xgb.plot_importance()
plt.figure(figsize=(30,20))
#F1 score is the combination of precision and recall: 2*((p*r) / (p+r))
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['axes.grid'] = False
#Performing cross validation on with special Dmatrix
params = {'objective': 'binary:logistic',
          'seed': 123,
          'eval_metric': 'auc'}

DTrain = xgb.DMatrix(X_train, label=y_train)
cv_df = xgb.cv(params, DTrain, num_boost_round = 5, early_stopping_rounds = 10, nfold = 5)
print(cv_df)

#Cross_val_score
cv_scores = cross_val_score(X_train, y_train, cv=5)

fallout_xg, sensitivity_xg, threshold_xg = roc_curve(y_test, xg_preds_proba)
plt.plot(fallout_xg, sensitivity_xg, color='blue')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('fallout')
plt.ylabel('sesitivity')
plt.show()

#Comparison between logistic regression and XgBoost

#Checking the macro average of f1 score
print(classification_report(y_test, preds_df['loan_status'], target_names = target_name))
print(classification_report(y_test, xg_preds, target_names = target_name))

#compare the roc auc score of the following
fallout_xg, sensitivity_xg, threshold_xg = roc_curve(y_test, xg_preds_proba)
plt.plot(fallout_xg, sensitivity_xg, color='blue')
fallout, sesitivity, thresholds = roc_curve(y_test, preds_df['prob_default'])
plt.plot(fallout, sesitivity, color='orange')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('fallout')
plt.ylabel('sesitivity')

auc_xg = roc_auc_score(y_test, xg_preds_proba)
auc_lr = roc_auc_score(y_test, preds_df['prob_default'])
print("Logistic Regression AUC Score: %0.2f" % auc_lr)
print("XG Boost AUC Score: %0.2f" % auc_xg)

#Optimization using Credit Acceptance rate
#Acceptance Rate = What percentage of new loans are to be accepted to keep the portfolio low

threshold = np.quantile(xg_preds, 0.85)
xg_pred_df['pred_loan_status'] = test_pred_df['prob_default'].apply(lambda x: 1 if x > threshold_85 else 0)
#It means we will accept 85 percent of the total loans and for this the maximum prob of default will be threshold

#Preparing strategy table to trade off between acceptance rate, threshold and bad rate
accept_rate = np.arange(1.0, 0.0, -0.05)
thresholds = []
bad_rates = []

for rate in accept_rate:
    threshold = np.quantile(xg_preds, rate)
    thresholds.append(threshold)
    xg_pred_df['pred_loan_status'] = xg_preds.apply(lambda x: 1 if x > threshold else 0)
    
    bad_rate = np.sum(xg_pred_df['true_loan_status']) / len(xg_pred_df['true_loan_status'])
    bad_rates.append(bad_rate)
    
print(test_pred_df.head())

# Calculate the bank's expected loss and assign it to a new column
test_pred_df['expected_loss'] = test_pred_df['prob_default'] * test_pred_df['loss_given_default'] * test_pred_df['loan_amnt']

# Calculate the total expected loss to two decimal places
total_exp_loss = round(np.sum(test_pred_df['expected_loss']),2)

# Print the total expected loss
print('Total expected loss: ', '${:,.2f}'.format(total_exp_loss))

#Gradient boosting portfolio calculation
portfolio = pd.DataFrame({'lr_prob':preds_df['prob_default'], 'xg_prob':xg_preds_proba, 'loan_amnt':X_test['loan_amnt'].reset_index(drop=True)})
portfolio['lgd'] = np.array(0.2)

portfolio = 




































