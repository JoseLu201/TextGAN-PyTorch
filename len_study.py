import numpy as np
import os
import matplotlib.pyplot as plt

from utils.data_loader import get_tokenlized_words

path = '/home/joselu/TFG/TextGAN-PyTorch/data_process/partidos_final'


def get_file_paths(root_dir):
    """
    Recursively iterates through a directory tree and returns a list of absolute file paths.

    Args:
        root_dir: The root directory to start the search from.

    Returns:
        A list of absolute file paths.
    """

    file_paths = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt') and file.startswith('orig_'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    return file_paths


# Example usage:

all_files = get_file_paths(path)

your_tweets = []
for file_path in all_files:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            your_tweets.append(line.strip())


print(len(your_tweets))
print(type(your_tweets))
print(your_tweets[:5])

your_tweets = get_tokenlized_words(your_tweets)
# Example list of tweet lengths in tokens
# print(your_tweets[:5])
tweet_lengths = [len(tweet) for tweet in your_tweets]


# Calculate statistics
mean_length = np.mean(tweet_lengths)
median_length = np.median(tweet_lengths)
percentile_95 = np.percentile(tweet_lengths, 95)
percentile_99 = np.percentile(tweet_lengths, 99)

print(f"Mean tweet length: {mean_length}")
print(f"Median tweet length: {median_length}")
print(f"95th percentile tweet length: {percentile_95}")
print(f"99th percentile tweet length: {percentile_99}")


# Manually set the number of bins
num_bins = 20  
counts, bins, patches = plt.hist(tweet_lengths, bins=num_bins, density=True)

# Calculate bin centers
bin_centers = (bins[:-1] + bins[1:]) / 2

# Set x-axis tick labels to show fewer labels
plt.xticks(bin_centers[::2], bin_centers[::2].astype(int), ha='right')  # Show every second bin label

plt.xlabel('Longitud del Tweet (palabras)', fontsize=12, fontweight='bold')
plt.ylabel('Densidad', fontsize=12, fontweight='bold')
plt.title('Distribución de la Longitud de los Tweets', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)

# Save the figure with high quality (optional)
plt.savefig('histograma.png', dpi=300)

# Display the plot
plt.show()
