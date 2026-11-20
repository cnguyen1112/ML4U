from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
# Array include 2 feature 1. studyHours, 2. numberofSession
featureArray = [
    [2, 3], [3, 4], [1, 2], [4, 3],   
    [7, 8], [8, 9], [6, 7], [9, 10]] # this is default DataSet
y = [0, 0, 0, 0, 1, 1, 1, 1]
model = DecisionTreeClassifier(criterion= 'entropy', random_state= 42)
model.fit(featureArray, y)

newStudents = [[6,8],[1,16]]
predicted = model.predict(newStudents)
print("Result", predicted[1])
plt.figure(figsize=(8,5))
plot_tree(model, feature_names=("Study Hours", "Number of Sessions"), class_names= ("Fail", "Pass"), filled= True)
plt.show()