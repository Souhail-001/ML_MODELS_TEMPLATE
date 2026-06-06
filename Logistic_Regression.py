import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train = pd.read_csv("")
test = pd.read_csv("")
print(train.columns)
print(test.columns)

y = train['class']

test_ids = test["id"]
test = test.drop(columns=["id"])

X = train.drop(columns=['class', 'id'])

num_cols = X.select_dtypes(include=['number']).columns
cat_cols = X.select_dtypes(include=['object']).columns

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X[num_cols] = scaler.fit_transform(X[num_cols])
test[num_cols] = scaler.transform(test[num_cols])

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])


X = pd.get_dummies(X, drop_first=True)
num_cols = test.select_dtypes(include=['number']).columns
cat_cols = test.select_dtypes(include=['object']).columns

for col in num_cols:
    test[col] = test[col].fillna(test[col].median())

for col in cat_cols:
    test[col] = test[col].fillna(test[col].mode()[0])

test = pd.get_dummies(test, drop_first=True)

test = test.reindex(columns=X.columns, fill_value=0)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=1.0,
    max_iter=5000,
    random_state=42
)

#from sklearn.model_selection import cross_val_score,StratifiedKFold

#skf = StratifiedKFold(
#    n_splits=5,
#    shuffle=True,
#    random_state=42
#)
#
#scores = cross_val_score(
#    model,
#    X,
#    y,
#    cv=skf,
#    scoring="roc_auc"
#)

model.fit(X_train, y_train)

from sklearn.metrics import balanced_accuracy_score

valid_preds = model.predict(X_test)

score = balanced_accuracy_score(
    y_test,
    valid_preds
)

print(score)

preds = model.predict(test)

preds = le.inverse_transform(preds.astype(int))

submission = pd.DataFrame({
    "id": test_ids,
    "class": preds
})

submission.to_csv("submission.csv", index=False)

submission.shape