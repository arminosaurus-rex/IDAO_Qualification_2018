# -*- coding: utf-8 -*-


# -- ==createFeatures== --

def createFeatures(user, day, userId3Visits, id3s, duration=7, sums=False):
    id3Visited = []
    for d in range(max(0, day - duration), day):
        if (user, d) in userId3Visits:
            id3Visited += userId3Visits[(user, d)]
    return [id3Visited.count(i) for i in id3s] + ([len(id3Visited)] if sums else [])

# -- ==createFeatures== --


# -- ==findX1Samples== --

def findX1Samples(id3s, userId3Visits, minDayTrain, maxDayTrain, users, duration, cutoff=10000, verbose=False):
    X1 = {id3: [] for id3 in id3s}
    for i, user in enumerate(users):
        if verbose and i % 100 == 0: print("user {} of {}".format(i, len(users)))
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
        if verbose and i % 100 == 0: print("id3 {} of {}".format(i, len(id3s)))
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
        if verbose and i % 100 == 0: print("regressor {} of {}".format(i, len(id3s)))
        if len(X0[id3]) > 0 and len(X1[id3]) > 0:
            regressors[id3] = RandomForestRegressor(max_depth=3, n_estimators=3).fit(X0[id3] + X1[id3], [0] * len(X0[id3]) + [1] * len(X1[id3])) 
    return regressors

# -- ==createRegressors== --



# -- ==computePredictions== --

def computePredictions(users, regressors, maxDayTrain, userId3Visits, duration, id3s, verbose=False):
    predictions = {user: {} for user in users}
    for i, user in enumerate(users):
        if verbose and i % 1000 == 0: print("user {} of {}".format(i, len(users)))
        prof = createFeatures(user, maxDayTrain, userId3Visits, id3s, duration=duration, sums=True)
        visited = sum([userId3Visits[(user, d)] if (user, d) in userId3Visits else [] for d in range(max(0, maxDayTrain - 21), maxDayTrain)], [])
        for id3, regressor in regressors.items():
            if not id3 in visited:
                predictions[user][id3] = regressor.predict([prof])[0]
    return predictions

# -- ==computePredictions== --



# -- ==extractTopPredictions== --

def extractTopPredictions(predictions, users, topCount=-1, verbose=False):
    #sum up
    if topCount == -1:
        topCount = len(predictions) // 20
    #regressorScores = {id3: regressors[id3].score(X0[id3] + X1[id3], [0] * len(X0[id3]) + [1] * len(X1[id3])) for id3 in id3s}
    predictedId3s = {user: sorted(predictions[user].keys(), key=lambda id3: -predictions[user][id3])[:5] for user in users}
    certainty = {user: sum([predictions[user][id3] for id3 in predictedId3s[user]]) for user in users}
    topUsers = sorted(userToId3.keys(), key=lambda user: -certainty[user])[:topCount]
    
    #create df
    dfData = {"user_id": topUsers}
    for i in range(5):
        dfData["id3_{}".format(i+1)] = [predictedId3s[user][i] for user in topUsers]
    ret = pd.DataFrame(data=dfData)
    if verbose: print(ret)
    return ret

# -- ==extractTopPredictions== --



# -- ==predict== --

def predict(train, trainUsers=1000, verbose=False, duration=2):
    id3s = train["id3"].unique()
    users = train["user_id"].unique()
    minDayTrain = train["date"].min()
    maxDayTrain = train["date"].max()
    userId3Visits = userVisits(train)
    X1 = findX1Samples(id3s, userId3Visits, minDayTrain, maxDayTrain, users[:trainUsers], duration, cutoff=1000, verbose=verbose)
    X0 = findX0Samples(X1, id3s, verbose=verbose)
    regressors = createRegressors(X0, X1, id3s, verbose=verbose)
    predictions = computePredictions(users, regressors, maxDayTrain, userId3Visits, duration, id3s, verbose=verbose)
    df = extractTopPredictions(predictions, users, verbose=verbose)
    return df

# -- ==predict== --


# -- ==userVisits== --

from ediblepickle import checkpoint

@checkpoint(work_dir="data/processed", key=lambda args, kwargs: "userDayVisits")
def userVisits(df):
    userId3Visits = df.groupby(["user_id", "date"])["id3"].apply(lambda x: list(set(x))).to_dict()
    return userId3Visits

# -- ==userVisits== --