# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 08:56:27 2023

@author: owfin
"""

import pandas as pd
import polars as pl #alternative to pandas with good options for manipulating dataframes
import numpy as np

#%% creating vars

dir_read = 'E:/(E)General Assembly stuff/Capstone'

years = [i for i in range(2008, 2021)]
modes = ("train", "test")
sentiments = ("neg", "pos")

paths = {}
for i in years:
    for ii in sentiments:
        paths[f"{i}/mode/{ii}"] = [i, "", ii, []]


del(i, ii)
#%% defining function to apply to each chunk

def write_to_bin(reader, path_dictionary:dict):
    
    paths_ = list(path_dictionary.keys()) #temporary struct to elim bins from rotation as they're filled
    
    for chunk in reader:
        df = chunk.copy()
        
        df['sentiment'] = np.where(df['stars'] < 3, "neg", "pos")
        df['year'] = df['date'].dt.year
        
        df = df[['text', 'sentiment', 'year']]
        
        for p in paths_:
            pf = pl.from_pandas(df).filter(
                (pl.col("sentiment") == path_dictionary[p][2]) & (pl.col("year") == path_dictionary[p][0])
                )
            
            dist = 25_000 - len(path_dictionary[p][3])
            
            if dist - len(pf['text']) >= 0:
                path_dictionary[p][3].extend(list(pf.select('text'))[0])
            else:
                path_dictionary[p][3].extend(list(pf.select('text'))[0][:dist])
                paths_.remove(p)
                print(f"{p} \n bin has been filled! ({26 - len(paths_)}/26 done)")
        
        if sum([len(path_dictionary[i][3]) for i in \
            path_dictionary.keys()]) == (25_000 * len(path_dictionary.keys())):
                break
        
    print("Done!")
    return

#%% applying function to thing

jreader =  pd.read_json\
    (dir_read + "/Final/yelp_academic_dataset_review.json", lines=True, chunksize=100_000)
        
write_to_bin(jreader, paths)

#%% see how many files each bin got

for p in paths.keys():
    print(f"Path {p} has:\n   {len(paths[p][3])} files!")

del(p)

#%%
# =============================================================================
# 
# Okay, we're done collecting data from the json. I ended up having to read it with pandas so
# that I could chunk it properly. Then I switched to polars so that I could manipulate it better.
#
# All in all, this took like ten minutes to run (!!!!). 
# 
# Now I'm on to writing these strings to text files in the proper file structure and compacting that
# structure so it's easier to load into Colab. Exciting things happening!
#
# =============================================================================

#%% imports

import os
import tarfile

#%% creating file paths

dir_write = 'G:/My Drive/Capstone'

for p in paths.keys():
    for mode in modes:
        new_path = os.path.join(dir_read, "keras", p.replace("mode", mode))
        if not os.path.exists(new_path):
            os.makedirs(new_path)

os.listdir(dir_write)

del(p, mode, new_path)
#%% for bins containing 25,000 files, send half to the training file and half to the testing file

for p in paths.keys():
    
    if len(paths[p][3]) == 25_000:
        
        for i in range(0, 12_500):
            path = os.path.join(dir_read, "keras", p.replace("mode", "train"))
            try:
                with open(path + f"/{i}.txt", "x", encoding="utf-8") as f:
                    f.write(paths[p][3][i])
            except FileExistsError:
                pass
                
        for i in range(12_500, 25_000):
            path = os.path.join(dir_read, "keras", p.replace("mode", "test"))
            try:
                with open(path + f"/{i}.txt", "x", encoding="utf-8") as f:
                    f.write(paths[p][3][i])
            except FileExistsError:
                pass
                    
        print(f"Path {p}\n\t has been filled!")
    else:
        print(f"Path {p}\n\t does NOT have enough content!")

del(p, i, path)
#%% compacting the whole file tree into a .tar.gz file
tar_name = 'cap_reviews.tar.gz'

tar_path = os.path.join(dir_write, tar_name)
tar = tarfile.open(tar_path, "w:gz")

for p in paths.keys():
    
    if len(paths[p][3]) == 25_000:
        
        path = os.path.join(dir_read, "keras", p.replace("mode", "test"))
        for text in os.listdir(path):
            tar.add(f"{path}/{text}", arcname=f"keras/{p.replace('mode', 'test')}/{text}.txt")
        
        path = os.path.join(dir_read, "keras", p.replace("mode", "train"))
        for text in os.listdir(path):
            tar.add(f"{path}/{text}", arcname=f"keras/{p.replace('mode', 'train')}/{text}.txt")
        
        print(f"Path {p}\n\t is finished processing!")
    
    else:
        print(f"Path {p} doesn't have enough files!")

tar.close()

del(p, text)
#%%
# =============================================================================
# 
# Now that that the file is created, I'm going to collect some metadata about the reviews themselves.
# 
# Actually I'm just going to write the data to a csv and maybe try to do some analysis on that file.
# 
# =============================================================================
#%%
cap_reviews = {}
for p in paths.keys():
    cap_reviews[p] = paths[p][3]
    while len(cap_reviews[p]) < 25_000:
        cap_reviews[p].extend(np.zeros(25_000 - len(cap_reviews[p]), dtype=int))

cap_reviews_lf = pl.DataFrame(cap_reviews).lazy()


#%% 
os.chdir(dir_read)

cap_reviews_lf.collect().write_csv("cap_reviews.csv")


