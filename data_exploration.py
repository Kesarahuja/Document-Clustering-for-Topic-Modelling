
import os
from sklearn.datasets import load_files

# Define the path to the dataset
data_path = '/home/ubuntu/twenty_newsgroups/20_newsgroups'

# Load the dataset
# categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med'] # Example of specific categories
newsgroups_data = load_files(data_path, encoding='latin1', random_state=42)

# Print some basic information about the dataset
print(f"Number of samples: {len(newsgroups_data.data)}")
print(f"Number of categories: {len(newsgroups_data.target_names)}")
print("Categories:")
for i, category in enumerate(newsgroups_data.target_names):
    print(f"  {i}: {category}")

# Print the first document from the first category as an example
print("\n--- Example Document ---")
print(newsgroups_data.data[0])

# Save categories and a sample document to a file for review
with open('/home/ubuntu/twenty_newsgroups/dataset_info.txt', 'w') as f:
    f.write(f"Number of samples: {len(newsgroups_data.data)}\n")
    f.write(f"Number of categories: {len(newsgroups_data.target_names)}\n")
    f.write("Categories:\n")
    for i, category in enumerate(newsgroups_data.target_names):
        f.write(f"  {i}: {category}\n")
    f.write("\n--- Example Document ---\n")
    f.write(newsgroups_data.data[0])

print("Dataset information saved to /home/ubuntu/twenty_newsgroups/dataset_info.txt")


