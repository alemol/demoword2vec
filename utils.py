# -*- coding: utf-8 -*-
#
# utilities
#
# Created by Alex Molina
# July 2020

import os
import pandas as pd
import gensim

def text_generator(dir_name, tok_and_tag=False):
    """text generator reads from csv files yielding str"""
    for root, directory, files in os.walk(dir_name, topdown=True):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                data_table = pd.read_csv(full_path,
                    encoding='UTF-8',
                    engine='python')
                data = pd.DataFrame(data_table, columns = ['text'])
                for  i, t in enumerate(data.text):
                    try:
                        if tok_and_tag:
                            tokens = gensim.utils.simple_preprocess(t)
                            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
                        else:
                            yield t
                    except:
                        continue

def common_affix_coverage(ref_set, eval_set):
    overlap = set(ref_set) & set(eval_set)
    return float(len(overlap)) / len(ref_set)

