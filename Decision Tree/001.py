# 1. Choosing Dataset, in this question we chose Iris dataset, which is included in sklearn.datasets
# 2. Splitting Dataset into 2 parts, train and test. (if input has 2 sets include 1 train sets and 1 test sets-> skip)
# 3. Model Selection & Training. (in this question we use DecisionTree).
# 4. Evaluating Accuracy Score for Model that we trained. 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
iris = load_iris()
featureArray = iris.data
y = iris.target

featureArray_train, featureArray_test, y_train, y_test = train_test_split(featureArray, y, test_size= 0.2, random_state= 42)


for depth in [1, 2, 3, None]:
    model = DecisionTreeClassifier(criterion='entropy', max_depth= depth, random_state= 42)
    model.fit(featureArray_train, y_train)

    y_train_pred = model.predict(featureArray_train)
    y_test_pred = model.predict(featureArray_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"max_depth = {str(depth):<4} | Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")

plt.figure(figsize=(32,20))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled= True, fontsize= 8 )
plt.show()



