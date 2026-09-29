#: Implementation of Decision Tree Learning using ID3 Algorithm 
import math

def entropy(data):
    """Calculates the Shannon entropy of a dataset based on its target labels."""
    labels = [row[-1] for row in data]
    total = len(labels)
    
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

def split_data(data, attribute_index, value):
    """Splits the dataset on a specific attribute value and removes that attribute column."""
    return [
        row[:attribute_index] + row[attribute_index+1:] 
        for row in data 
        if row[attribute_index] == value
    ]

def info_gain(data, attribute_index):
    """Calculates the Information Gain from splitting the data on a given attribute index."""
    total_entropy = entropy(data)
    values = set(row[attribute_index] for row in data)
    total = len(data)
    
    sub_entropy = 0.0
    for val in values:
        subset = [row for row in data if row[attribute_index] == val]
        sub_entropy += (len(subset) / total) * entropy(subset)
        
    return total_entropy - sub_entropy

def id3(data, features):
    """
    Recursively builds a decision tree using the ID3 algorithm.
    
    Arguments:
    data     -- A list of lists representing rows of data (target label is the last element).
    features -- A list of strings containing the names of the remaining attributes.
    """
    labels = [row[-1] for row in data]
    
    # Base Case 1: If all examples are pure (have the same label), return that label.
    if len(set(labels)) == 1:
        return labels[0]
        
    # Base Case 2: If no features are left to split on, return the majority label.
    if len(features) == 0:
        return max(set(labels), key=labels.count)
        
    # Choose the attribute with the highest Information Gain
    best_feat_idx = max(range(len(features)), key=lambda i: info_gain(data, i))
    best_feature = features[best_feat_idx]
    
    # Initialize the sub-tree rooted at the chosen feature
    tree = {best_feature: {}}
    
    # Remove the chosen feature from the feature list for downstream branches
    remaining_features = [f for i, f in enumerate(features) if i != best_feat_idx]
    
    # Split the dataset for each unique value of the best feature
    feature_values = set(row[best_feat_idx] for row in data)
    for val in feature_values:
        subset = split_data(data, best_feat_idx, val)
        tree[best_feature][val] = id3(subset, remaining_features)
        
    return tree


# --- Example Execution ---
if __name__ == "__main__":
    # Standard PlayTennis Subset Dataset
    dataset = [
        ['Sunny', 'Hot', 'High', 'Weak', 'No'],
        ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'Yes']
    ]
    features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    
    decision_tree = id3(dataset, features)
    
    import pprint
    print("Generated ID3 Decision Tree:")
    pprint.pprint(decision_tree)
