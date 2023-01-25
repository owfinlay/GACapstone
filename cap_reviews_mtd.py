# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 05:45:11 2023

@author: owfin
"""

import polars as pl
import os
import re

#%%

path = os.path.abspath('E:/(E)General Assembly stuff/Capstone/Final')
filename = 'cap_reviews.csv'

filepath = os.path.join(path, filename)

df = pl.read_csv(filepath)

del(path, filename, filepath)
#%%

neg_dic = {}
pos_dic = {}

for i in df.columns[::2]:
    neg_dic[i[:4]] = df[i]
for i in df.columns[1::2]:
    pos_dic[i[:4]] = df[i]

pos_df = pl.DataFrame(pos_dic)
neg_df = pl.DataFrame(neg_dic)

better_df = pos_df.vstack(neg_df)

del(i, neg_dic, pos_dic, neg_df, pos_df)
#%% general info

mtd = {}

html_pattern = "<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>" #from https://uibakery.io/regex-library/html-regex-python
alph_pattern = re.compile(r"[A-Za-z]") # both from stackoverflow
upper_pattern = re.compile(r"[A-Z]")

for i in df.columns:
    col_Ser = df.select(i).get_columns()[0]
    col_as_string = col_Ser.str.concat("")[0]
    mtd[i] = {
        'path'          : [i],
        'is_not_full'   : [col_Ser.is_null().any()],
        'mean_len'      : [col_Ser.str.lengths().mean()],
        'median_len'    : [col_Ser.str.lengths().median()],
        'std_len'       : [col_Ser.str.lengths().std()],
        'html_per_K'    : [len(col_Ser.str.concat("").str.count_match(html_pattern))/25],
        'pct_upper'     : [len(re.findall(upper_pattern, col_as_string)) \
            / len(re.findall(alph_pattern, col_as_string))],
        'pct_nums'      : [len(re.findall('[0-9]', col_as_string)) \
            / len(col_as_string)],
        'pct_!alnumeric': [len(re.findall('\W', col_as_string)) / len(col_as_string)]
        }

#%% every year's favorite words

from nltk.corpus import stopwords


most_common_words = {}
stop_words = set(stopwords.words('english'))

for i in better_df.columns:
    col_Ser = better_df.select(i).get_columns()[0]
    col_as_string = col_Ser.str.concat(" ")
    
    most_common_words[i + "_SW"] = col_as_string.str.split(by=" ")[0]\
        .value_counts(sort=True).head(100)
    most_common_words[i+"_SW"].columns = [i+'_SW', i+'_SW_ct']
    
    
    most_common_words[i + "_no_SW"] = col_as_string.str.split(by=" ")[0]\
        .value_counts(sort=True).filter(
            ~pl.col(i).str.to_lowercase().is_in(list(stop_words)+["","null"])
            ).head(100)
    most_common_words[i + "_no_SW"].columns = [i+"_no_SW", i+"_no_SW_ct"]



#%%

os.chdir("E:/(E)General Assembly stuff/Capstone")

import pandas as pd

mtd_df = pd.DataFrame.from_dict(mtd, orient='index')
mtd_df.to_csv("reviews_metadata")

better_df.write_csv("yearly_reviews")

mcw = pl.DataFrame(range(1, 101))

for i in list(most_common_words.keys()):
    mcw = mcw.hstack(most_common_words[i])
        
mcw.write_csv("most_common_words")

cmn_words = pd.DataFrame(most_common_words)




