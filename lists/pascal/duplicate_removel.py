import os
from pathlib import Path 

root_path = Path("/share/home/orfu/DeepLearning/GithubModels/BAM/lists/pascal")
raw_path = root_path / 'voc_sbd_merge.txt'
new_path = root_path / 'voc_sbd_merge_noduplicate.txt'
lines = open(raw_path).readlines()
new_f = open(new_path, 'w+')


existing_lines = []
for line in lines:
    if line not in existing_lines:
        existing_lines.append(line)
        new_f.write(line)
print('Ori: {}, new: {}'.format(len(lines), len(existing_lines)))
print('Finished.')