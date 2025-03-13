from sklearn.datasets import load_iris
x,y=load_iris(return_X_y=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.33, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)
score=model.score(x_test,y_test)
print(score)