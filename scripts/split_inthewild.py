"""Split official in the wild dataset into real and fake audio files."""
import csv
import os
import shutil

spoof_files = []
real_files = []
path = "./data/inthewild/release_in_the_wild"

if not os.path.exists(f"{path}/real/"):
    os.mkdir(f"{path}/real/")
if not os.path.exists(f"{path}/fake/"):
    os.mkdir(f"{path}/fake/")

with open(f"{path}/meta.csv", "r") as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        if row[-1] == "spoof":
            spoof_files.append(row[0])
        elif row[-1] == "bona-fide":
            real_files.append(row[0])

print(len(spoof_files))
print(len(real_files))

for file_path in real_files:
    shutil.move(f"{path}/{file_path}", f"{path}/real/")

for file_path in spoof_files:
    shutil.move(f"{path}/{file_path}", f"{path}/fake/")
