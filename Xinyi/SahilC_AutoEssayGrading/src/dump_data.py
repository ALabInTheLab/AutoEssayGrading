# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 17:24:57 2016

@author: Pranav
"""

import python-utils

training_data_file = '../dataset/small_training_set.tsv'
test_data_file = '../dataset/small_test_set.tsv'

training_data_dump = '../dumps/training_data_dump'
test_data_dump = '../dumps/test_data_dump'

training_data = python_utils.get_data(training_data_file)
python_utils.dump_object(training_data, training_data_dump)

test_data = python_utils.get_data(test_data_file)
python_utils.dump_object(test_data, test_data_dump)
