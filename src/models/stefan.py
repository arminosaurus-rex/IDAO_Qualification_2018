# -*- coding: utf-8 -*-


# -- ==createFeatures== --

def createFeatures(user, day, userId3Visits, id3s, duration=7, sums=False):
    id3Visited = []
    ret = []
    for d in range(day, day - duration, -1):
        if (user, d) in userId3Visits:
            id3Visited += userId3Visits[(user, d)]
        ret += [id3Visited.count(i) for i in id3s]
        if sums: ret.append(len(id3Visited))
    return ret

# -- ==createFeatures== --


# -- ==findX1Samples== --

def findX1Samples(id3s, userId3Visits, minDayTrain, maxDayTrain, users, duration, cutoff=10000, verbose=False):
    X1 = {id3: [] for id3 in id3s}
    for i, user in enumerate(users):
        if verbose: reportProgress("computation of X1 samples", i, len(users))
        for day in range(maxDayTrain, minDayTrain - 1, -1):
            feat = createFeatures(user, day, userId3Visits, id3s, duration=duration, sums=True)
            for id3 in id3s:
                if len(X1[id3]) < cutoff and (user, day) in userId3Visits and id3 in userId3Visits[(user, day)]:
                    if not any([(user, d) in userId3Visits and id3 in userId3Visits[(user, d)] for d in range(max(0, day - 21), day)]):
                        X1[id3].append(feat)
    return X1

# -- ==findX1Samples== --



# -- ==findX0Samples== --

from random import shuffle

def findX0Samples(X1, id3s, verbose=False):
    X0 = {}
    for i, id3 in enumerate(id3s):
        if verbose: reportProgress("computation of X0 samples", i, len(id3s))
        rows = sum([X1[i] for i in id3s if not i == id3], [])
        shuffle(rows)
        X0[id3] = rows[:len(X1[id3])]
    return X0

# -- ==findX0Samples== --



# -- ==createRegressors== --

from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor

def createRegressors(X0, X1, id3s, verbose=False):
    regressors = {}
    for i, id3 in enumerate(id3s):
        if verbose: reportProgress("fitting regressors", i, len(id3s))
        if len(X0[id3]) > 2 and len(X1[id3]) > 2:
            regressors[id3] = RandomForestRegressor(max_depth=3, n_estimators=3).fit(X0[id3] + X1[id3], [0] * len(X0[id3]) + [1] * len(X1[id3])) 
    return regressors

# -- ==createRegressors== --



# -- ==computePredictions== --

def computePredictions(users, regressors, maxDayTrain, userId3Visits, duration, id3s, verbose=False):
    visited, profiles = {}, []
    for i, user in enumerate(users):
        if verbose: reportProgress("computing eligible id3s", i, len(users))
        profiles.append(createFeatures(user, maxDayTrain, userId3Visits, id3s, duration=duration, sums=True))
        visited[user] = sum([userId3Visits[(user, d)] if (user, d) in userId3Visits else [] for d in range(max(0, maxDayTrain - 21), maxDayTrain)], [])
        
    predictions = {user: {} for user in users}
    for i, id3 in enumerate(id3s):
        if verbose: reportProgress("predicting user behaviour", i, len(id3s))
        if id3 in regressors:
            tpred = regressors[id3].predict(profiles)
            for i, user in enumerate(users):
                if not id3 in visited[user]:
                    predictions[user][id3] = tpred[i]
    return predictions

# -- ==computePredictions== --



# -- ==extractTopPredictions== --

import pandas as pd

def extractTopPredictions(predictions, users, topCount=-1, verbose=False):
    #sum up
    if topCount == -1:
        topCount = len(predictions) // 20
    #regressorScores = {id3: regressors[id3].score(X0[id3] + X1[id3], [0] * len(X0[id3]) + [1] * len(X1[id3])) for id3 in id3s}
    predictedId3s = {user: sorted(predictions[user].keys(), key=lambda id3: -predictions[user][id3])[:5] for user in users}
    certainty = {user: sum([predictions[user][id3] for id3 in predictedId3s[user]]) for user in users}
    topUsers = sorted(users, key=lambda user: (len(predictions[user]) < 5, -certainty[user]))[:topCount]
    
    #create df
    dfData = {"user_id": topUsers}
    for i in range(5):
        dfData["id3_{}".format(i+1)] = [predictedId3s[user][i] for user in topUsers]
    ret = pd.DataFrame(data=dfData)
    if verbose: print(ret)
    return ret

# -- ==extractTopPredictions== --



# -- ==predict== --

def predict(train, trainUsers=1000, verbose=False, duration=5, minImpressionsUsers=100, minImpressionsId3s=100000, cutoff=25000):
    #find id3s and users
    if verbose: print("finding users and id3s...")
    id3s = train["id3"].unique()
    users = train["user_id"].unique()
    
    #apply request filter
    if verbose: print("applying request filter...")
    vcUsers = train["user_id"].value_counts()
    vcId3s = train["id3"].value_counts()
    activeUsers = list(vcUsers[vcUsers >= minImpressionsUsers].index)
    activeId3s = list(vcId3s[vcId3s >= minImpressionsId3s].index)
    if len(activeUsers) < len(users) // 20: print("WARNING: less than 5% active users, can't predict 5%")
    train = train[train["user_id"].isin(activeUsers) & train["id3"].isin(activeId3s)]
    if verbose: print("{} users ({} before filter), {} id3s ({} before filter), {} rows".format(len(activeUsers), len(users), len(activeId3s), len(id3s), train["id3"].count()))
    
    #compute lookup tables
    if verbose: print("computing lookup tables...")
    minDayTrain = train["date"].min()
    maxDayTrain = train["date"].max()
    userId3Visits = userVisits(train)
    
    #predict
    X1 = findX1Samples(activeId3s, userId3Visits, minDayTrain, maxDayTrain, activeUsers[:trainUsers], duration, cutoff=cutoff, verbose=verbose)
    X0 = findX0Samples(X1, activeId3s, verbose=verbose)
    regressors = createRegressors(X0, X1, activeId3s, verbose=verbose)
    predictions = computePredictions(activeUsers, regressors, maxDayTrain, userId3Visits, duration, activeId3s, verbose=verbose)
    df = extractTopPredictions(predictions, activeUsers, topCount=max(len(activeUsers), len(users) // 20), verbose=verbose)
    return df

# -- ==predict== --


# -- ==userVisits== --

from ediblepickle import checkpoint

@checkpoint(work_dir="data/processed", key=lambda args, kwargs: "userDayVisits.{}rows".format(args[0]["id3"].count()))
def userVisits(df):
    userId3Visits = df.groupby(["user_id", "date"])["id3"].apply(lambda x: list(set(x))).to_dict()
    return userId3Visits

# -- ==userVisits== --


# -- ==reportProgress== --

import time
import datetime

progressInfo = {}

def formatTime(seconds):
    return "{}m {}s".format(round(seconds) // 60, round(seconds) % 60)

def reportProgress(name, current, count, updateFrequency=15):
    if not name in progressInfo:
        progressInfo[name] = {"lastUpdate": time.time(), "start": time.time(), "lastCount": current}
        print("starting {}...".format(name))
    elif current >= count - 1:
        print("finished {}".format(name))
    elif time.time() - progressInfo[name]["lastUpdate"] >= updateFrequency:
        print("computing {}, {}% done, {} elapsed, {} remaining".format(name, round(100.0 * current / count), formatTime(time.time() - progressInfo[name]["start"]), formatTime((time.time() - progressInfo[name]["lastUpdate"]) / (current - progressInfo[name]["lastCount"]) * (count - current))))
        progressInfo[name]["lastUpdate"] = time.time()
        progressInfo[name]["lastCount"] = current
        

# -- ==reportProgress== --