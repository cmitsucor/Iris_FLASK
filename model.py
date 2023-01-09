from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pickle
from sklearn.tree import DecisionTreeClassifier 
import pandas as pd
import joblib
 

# Reading the Iris.csv file

df = pd.read_csv('iris.csv')

array = df.values
iris = load_iris()
clf = DecisionTreeClassifier()

sdasdasda.setdefault()addsitedir(
                                 
                                 sda
                                 dasdsa, known_paths=None)

b,mcm,b cbreak(
    mcbc vmbc
    
    cv.m,bcv.,b
)


X = array[:,0:4]
y = array[:,4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#model = clf.fit(X_train, y_train)

clf.fit(X_train, y_train)
#print("accuracy :" , clf.score(X_train,y_train))

print("Mode score: ", clf.score(X_train, y_train))
print("Test Accuracy: ", clf.score(X_test, y_test))


# save the model to disk
joblib.dump(clf, 'model.pkl')
#pickle.dump(model, open("model.pkl", "wb"))
