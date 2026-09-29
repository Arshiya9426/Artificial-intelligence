# Implementation of Naïve Bayes Classification 
class NaiveBayes:
    def __init__(self):
        """Initializes the Naive Bayes model parameters."""
        self.prior = {}
        self.conditional = {}
        self.classes = []

    def fit(self, data):
        """
        Trains the Naive Bayes classifier on text/categorical data.
        
        Arguments:
        data -- A list of lists representing rows of data (target label is the last element).
        """
        total_records = len(data)
        labels = [row[-1] for row in data]
        self.classes = list(set(labels))
        
        # --- Calculate Prior Probabilities P(Class) ---
        for c in self.classes:
            self.prior[c] = labels.count(c) / total_records
            self.conditional[c] = {}
            
        # --- Calculate Conditional Probabilities P(Feature | Class) ---
        num_features = len(data[0]) - 1
        for c in self.classes:
            sub_data = [row[:-1] for row in data if row[-1] == c]
            sub_total = len(sub_data)
            
            for f_idx in range(num_features):
                self.conditional[c][f_idx] = {}
                feature_vals = [row[f_idx] for row in data]
                unique_vals = set(feature_vals)
                
                for val in unique_vals:
                    count = sum(1 for row in sub_data if row[f_idx] == val)
                    # Laplace smoothing formula: (count + 1) / (sub_total + vocabulary_size)
                    self.conditional[c][f_idx][val] = (count + 1) / (sub_total + len(unique_vals))

    def predict(self, sample):
        """
        Predicts the class label for a single given instance.
        
        Arguments:
        sample -- A list of features (without the target label).
        """
        best_class = None
        best_prob = -1.0
        
        for c in self.classes:
            prob = self.prior[c]
            for f_idx, val in enumerate(sample):
                if val in self.conditional[c][f_idx]:
                    prob *= self.conditional[c][f_idx][val]
                else:
                    # Handle unseen feature values gracefully by applying smoothing default
                    unique_vals_count = len(self.conditional[c][f_idx])
                    sub_total = sum(1 for val in self.conditional[c][f_idx])
                    prob *= 1 / (sub_total + unique_vals_count)
                    
            if prob > best_prob:
                best_prob = prob
                best_class = c
                
        return best_class


# --- Example Execution ---
if __name__ == "__main__":
    # Categorical dataset tracking weather attributes and whether a game is played
    dataset = [
        ['Sunny', 'High', 'No'],
        ['Sunny', 'High', 'No'],
        ['Overcast', 'High', 'Yes'],
        ['Rainy', 'Normal', 'Yes'],
        ['Rainy', 'Normal', 'Yes']
    ]
    
    nb = NaiveBayes()
    nb.fit(dataset)
    
    test_instance = ['Sunny', 'Normal']
    prediction = nb.predict(test_instance)
    
    print(f"Test instance: {test_instance}")
    print(f"Predicted Class: {prediction}")
