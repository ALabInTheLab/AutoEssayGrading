# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 22:42:51 2016

@author: Pranav
"""
import python-utils
test_data_file = '../dataset/small_test_set.tsv'
training_data_file = '../dataset/small_training_set.tsv'

glove_training_data_dump = '../dumps/glove_training_data_dump'
glove_test_data_dump = '../dumps/glove_test_data_dump'

glove_training_data = python_utils.get_glove_data(training_data_file)
python_utils.dump_object(glove_training_data, glove_training_data_dump)

glove_test_data = python_utils.get_glove_data(test_data_file)
python_utils.dump_object(glove_test_data, glove_test_data_dump)