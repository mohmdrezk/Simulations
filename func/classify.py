# import support vector machine from sklearn    
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler

def classify(X_train, y_train, X_test, y_test):

    scaler = StandardScaler()
    clf=SVC(kernel='linear', C=1)

    scaler.fit_transform(X_train)
    scaler.transform(X_test)

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    
    return accuracy