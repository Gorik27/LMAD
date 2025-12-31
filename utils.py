import re


def numerical_sort_key(filename):
    # Extract numerical parts from the filename for sorting
    parts = re.findall(r'\d+|\D+', filename)
    return [int(part) if part.isdigit() else part for part in parts]