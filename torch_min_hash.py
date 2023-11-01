import torch
import numpy as np

# Define the number of hash functions and the size of the hash table
num_hash_functions = 10000
hash_table_size = 1000

# Generate random hash functions and a random permutation of indices
hash_functions = torch.randint(0, hash_table_size, (num_hash_functions,))
permutation = torch.randperm(num_hash_functions)

# Create a hash table filled with infinity values
hash_table = torch.full((num_hash_functions, hash_table_size), float('inf'))

def minhash(set_to_hash):
    # Create an empty signature with infinity values
    signature = torch.full((num_hash_functions,), float('inf'))
    
    # For each element in the set, calculate the hash values using hash functions
    for element in set_to_hash:
        # print(signature)
        # Convert the element to a unique integer (e.g., using its hash code)
        element_hash = hash(element)
        
        import code
        code.interact(local=locals())
        
        # Update the signature with minimum hash values
        signature = torch.min(signature, (element_hash ^ hash_functions) % hash_table_size)
    
    return signature[permutation]

# Example sets to compare
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Convert the sets to PyTorch tensors
set1_tensor = torch.tensor(list(set1))
set2_tensor = torch.tensor(list(set2))

# Compute MinHash signatures for the sets
signature1 = minhash(set1_tensor)
signature2 = minhash(set2_tensor)

# Estimate Jaccard similarity using MinHash signatures
jaccard_similarity = torch.sum(signature1 == signature2).item() / num_hash_functions

print(f"Estimated Jaccard Similarity: {jaccard_similarity}")
