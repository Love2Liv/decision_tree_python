import load_datasets
import DecisionTree
import time


# Charger/lire les datasets
train, train_labels, test, test_labels = load_datasets.load_iris_dataset(0.6)

# Initializer/instanciez le classifieur
classifier = DecisionTree.DecisionTree()

start_time = time.time()
# Entrainez votre classifieur
classifier.train(train, train_labels)
end_time = time.time()

# Tester votre classifieur
output = classifier.predict(test)

#print("Output: ", output)

#results
confusion_matrix = classifier.computeConfusionMatrix(test_labels, output)
print("Confusion Matrix: ", confusion_matrix)
test_accuracy = classifier.accuracyScore(confusion_matrix)
print("Accuracy: ", test_accuracy)
total_time = end_time - start_time
print("Time: ", total_time)
