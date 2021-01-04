import numpy as np
import math
import pprint
import random
from scipy import stats

class DecisionTree:

	def __init__(self, **kwargs):
		pass

	def getLabelPosition(self, labels, l):
		for i, r in enumerate(labels):
			if r == l:
				return i

	def combineTwoLists(self, list_1, list_2):
		list_3 = np.concatenate([list_1, list_2])
		return set(np.sort(list_3))

	# computes the confusion matrix
	def computeConfusionMatrix(self, test_labels, output):
		# get unique labels from the test set
		test_unique_labels = self.getUniqueValues(test_labels)
		# get unique values from the train set
		train_unique_labels = self.unique_labels
		'''
		Sometimes there are values that appear in the test set that did not appear in the train set.
		So I take the union of the two sets below.
		'''
		new_unique_labels_list = test_unique_labels
		if np.array_equal(test_unique_labels, train_unique_labels):
			new_unique_labels_list = self.combineTwoLists(test_unique_labels, train_unique_labels)
		num_test = len(new_unique_labels_list)

		#create confusion matrix table
		matrix_table = []
		for c in range(num_test):
			table_row = [0 for i in range(num_test)]
			matrix_table.append(table_row)

		for t_label, o_label in zip(test_labels, output):
			c_value = self.getLabelPosition(new_unique_labels_list, t_label)
			p_value = self.getLabelPosition(new_unique_labels_list, o_label)
			matrix_table[c_value][p_value] += 1

		confusion_matrix = np.array(matrix_table)
		return confusion_matrix

	def accuracyScore(self, confusion_matrix):
		correct_pred = 0
		total_pred = 0
		count_col = 0
		for row in confusion_matrix:
			count_row = 0
			for item in row:
				total_pred += item
				if count_row == count_col:
					correct_pred += item
				count_row += 1
			count_col += 1

		percent = correct_pred / total_pred
		return percent

	# returns unique values from a given list
	def getUniqueValues(self, my_list):
		list_0 = set(my_list)
		unique = np.array(list(list_0))
		unique = np.sort(unique)

		return unique

	def computeLabelsEntropy(self, labels):
		#get unique labels
		unique_labels = self.getUniqueValues(labels)
		num_total = labels.shape[0]
		total_entropy = 0
		#calculate entropy
		for label in unique_labels:
			num_label = np.count_nonzero(labels == label)
			total_entropy += - (num_label/num_total)*math.log2(num_label/num_total)

		return total_entropy

	def computeColumnImpurity(self, column, labels):
		unique_column_values = self.getUniqueValues(column)
		unique_labels_values = self.getUniqueValues(labels)
		num_total = len(column)

		# create division table for the column
		entropy_table = {}
		for value in unique_column_values:
			my_value_count = {}
			for label in unique_labels_values:
				value_count = 0
				for i, row in enumerate(column):
					if (row == value) and (labels[i] == label):
						value_count += 1
				my_value_count[label] = value_count
				entropy_table[value] = my_value_count

		#compute the impurity of the column
		impurity = 0
		for label in entropy_table:
			num_labels = sum([entropy_table[label][x] for x in entropy_table[label]])
			labels_total = 0
			entropy = 0
			for j, v in enumerate(entropy_table[label]):
				l_count = entropy_table[label][v]
				labels_total += l_count
				if l_count == 0:
					continue

				entropy += - (l_count / num_labels) * math.log2(l_count / num_labels)
			impurity += (labels_total/num_total) * entropy

		return impurity

	def computeGain(self, train, train_labels):
		num_columns = np.size(train, 1)
		#separate training data into idividual columns
		columns_list = [train[:, i] for i in range(num_columns)]
		#compute the entropy of the training sample
		train_sample_entropy = self.computeLabelsEntropy(train_labels)

		#compute the gain for each column
		all_columns_gains = []
		for i, column in enumerate(columns_list):
			# first obtain the impurity of the column
			impurity = self.computeColumnImpurity(column, train_labels)
			# then compute the gain of the column and save it to the list
			all_columns_gains.append(train_sample_entropy - impurity)

		return all_columns_gains

	def loadNextBranchArrays(self, train, train_labels, node_index, value):
		'''
		Creates a new train array (along with the associated labels,) where all members of the column
		at index "node_index" will have the value described by "value".
		'''
		new_train = []
		new_train_labels = []
		for i, row in enumerate(train):
			if row[node_index] == value:
				new_train.append(row)
				new_train_labels.append(train_labels[i])

		return np.array(new_train), np.array(new_train_labels)

	def getMaxIndex(self, my_list, prior_indexes):
		'''
		Returns the maximum index from the list, so long as it is not present in the prior_indexes list
		'''
		el = -99999999
		idx = -1
		for i, e in enumerate(my_list):
			if (e > el) and (i not in prior_indexes):
				el = e
				idx = i
		return idx

	def getArrayMode(self, my_list):
		'''
		Returns the most common item from the list
		'''
		m = stats.mode(my_list)
		return m[0][0]

	def createDecisionTree(self, train, train_labels,  prior_indexes, tree):
		num_features = len(train[0])
		# get the list of gains
		gains = self.computeGain(train, train_labels)
		# get the index of highest gain
		max_gain_index = self.getMaxIndex(gains, prior_indexes)
		# update the list of prior indexes
		prior_indexes.append(max_gain_index)
		if not tree:
			tree = {max_gain_index: {}}
		# set the column with the highest gain as (root) node
		node = train[:, max_gain_index]
		# get the distinct values of the node
		node_values = self.getUniqueValues(node)
		for value in node_values:
			# get arrays for the next branch of the tree
			sub_tree_train, sub_tree_labels = self.loadNextBranchArrays(train, train_labels, max_gain_index, value)
			# get the unique values from the next branches of the tree
			new_label_values = self.getUniqueValues(sub_tree_labels)
			# if there is only one label, then we've reached a leaf node
			if len(new_label_values) == 1:
				tree[max_gain_index][value] = new_label_values[0]
			else:
				if len(prior_indexes) >= num_features:
					tree[max_gain_index][value] = self.getArrayMode(new_label_values)
				else:
					tree[max_gain_index][value] = self.createDecisionTree(sub_tree_train, sub_tree_labels, prior_indexes, {})

		return tree

	def train(self, train, train_labels):
		#create the decision tree
		t = {}
		prior_indexes = []
		self.tree = self.createDecisionTree(train, train_labels, prior_indexes, t)
		self.unique_labels = self.getUniqueValues(train_labels)

		print("Decision Tree: ")
		pprint.pprint(self.tree)


	def predict(self, test):
		predictions = []
		for item in test:
			predictions.append(self.test(item, self.tree))

		return predictions

	def test(self, test, tree):
		for k in tree:
			value = test[k]
			if (value in tree[k]) and (type(tree[k][value]) is dict):
				return self.test(test, tree[k][value])
			elif value in tree[k]:
				return tree[k][value]
			else:
				return random.choice(self.unique_labels)
