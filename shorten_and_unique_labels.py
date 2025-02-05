def shorten_and_unique_labels(labels):
    if isinstance(labels, list):
        # Handle list input
        labels = [label[:30] for label in labels]  # Shorten to 30 characters
        unique_labels = []
        for label in labels:
            suffix = 1
            new_label = label
            while new_label in unique_labels:  # Ensure uniqueness
                new_label = label[:29] + str(suffix)
                suffix += 1
            unique_labels.append(new_label)
        return unique_labels

    elif isinstance(labels, dict):
        # Handle dictionary input
        keys = [key[:30] for key in labels.keys()]  # Shorten to 30 characters
        unique_labels = {}
        for original_key, value in labels.items():
            new_key = original_key[:30]
            suffix = 1
            while new_key in unique_labels:  # Ensure uniqueness
                new_key = original_key[:29] + str(suffix)
                suffix += 1
            unique_labels[new_key] = value
        return unique_labels

    else:
        raise TypeError("Input must be a list or dictionary")
