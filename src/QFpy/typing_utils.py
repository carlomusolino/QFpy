import numpy as np

# Define scalar types
__scalar_types = (int, float, complex, str, bool, bytes, type(None), np.generic)

# Function to check if an object is scalar
def is_scalar(obj):
    return isinstance(obj, __scalar_types)
