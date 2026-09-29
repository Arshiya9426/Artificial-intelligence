# Implementation of Candidate Elimination Algorithm 
def candidate_elimination(data):
    """
    Implements the Candidate Elimination Algorithm.
    
    Arguments:
    data -- A list of lists, where each inner list represents a training instance.
            The last element of each list is the classification label ('Yes' or 'No').
    
    Returns:
    S -- The final Specific Boundary.
    G -- The final General Boundary.
    """
    num_attributes = len(data[0]) - 1
    
    # Initialize Specific (S) and General (G) boundaries
    S = ['0'] * num_attributes
    G = [['?'] * num_attributes]
    
    # Initialize S with the first positive training example
    for row in data:
        if row[-1] == 'Yes':
            S = row[:-1].copy()
            break

    # Process all training examples
    for row in data:
        inputs, label = row[:-1], row[-1]
        
        if label == 'Yes':
            # --- Generalize S ---
            for i in range(num_attributes):
                if inputs[i] != S[i]:
                    S[i] = '?'
            
            # --- Prune G ---
            # Remove from G any hypothesis that is inconsistent with this positive example
            G = [
                g for g in G 
                if all(g[i] == '?' or g[i] == inputs[i] for i in range(num_attributes))
            ]
            
        else:
            # --- Specialize G ---
            G_new = []
            for g in G:
                # If the hypothesis already rejects this negative example, keep it as-is
                if not all(g[i] == '?' or g[i] == inputs[i] for i in range(num_attributes)):
                    G_new.append(g)
                else:
                    # Specialize g by replacing '?' with specific values from S
                    for i in range(num_attributes):
                        if g[i] == '?' and inputs[i] != S[i]:
                            g_candidate = g.copy()
                            g_candidate[i] = S[i]
                            
                            # Filter out redundant hypotheses
                            if g_candidate not in G_new:
                                G_new.append(g_candidate)
            
            # Update G boundary and prune hypotheses that are not more general than S
            G = [
                g for g in G_new 
                if all(g[i] == '?' or g[i] == S[i] for i in range(num_attributes))
            ]

    return S, G


# --- Example Execution ---
if __name__ == "__main__":
    # Standard EnjoySport Dataset
    dataset = [
        ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
    ]
    
    s_boundary, g_boundary = candidate_elimination(dataset)
    
    print("Final Specific Boundary (S):")
    print(s_boundary)
    print("\nFinal General Boundary (G):")
    print(g_boundary)
