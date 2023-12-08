import os
import sys
import shutil
from argparse import ArgumentParser

from datagen import DataGen

parser = ArgumentParser()
parser.add_argument('--src_data_dir', type=str, help='data directory')
parser.add_argument('--tar_data_dir', type=str, help='data directory')
parser.add_argument('--data_fn', type=str, help='data file that indexs all shape-ids')
parser.add_argument('--primact_types', type=str,
                    help='list all primacts [separated by comma], default: None, meaning all', default=None)
parser.add_argument('--category_types', type=str,
                    help='list all categories [separated by comma], default: None, meaning all', default=None)
parser.add_argument('--num_processes', type=int, default=40, help='number of CPU cores to use')
parser.add_argument('--num_epochs', type=int, default=200, help='control the data amount')
parser.add_argument('--starting_epoch', type=int, default=0,
                    help='if you want to run this data generation across multiple machines, you can set this parameter so that multiple data folders generated on different machines have continuous trial-id for easier merging of multiple datasets')
parser.add_argument('--out_fn', type=str, default=None,
                    help='a file that lists all valid interaction data collection [default: None, meaning data_tuple_list.txt]. Again, this is used when you want to generate data across multiple machines. You can store the filelist on different files and merge them together to get one data_tuple_list.txt')
conf = parser.parse_args()

if conf.out_fn is None:
    conf.out_fn = 'data_tuple_list.txt'

if conf.primact_types is None:
    conf.primact_types = ['pushing', 'pushing-up', 'pushing-left', 'pulling', 'pulling-up', 'pulling-left']
else:
    conf.primact_types = conf.primact_types.split(',')
print(conf.primact_types)

if conf.category_types is None:
    conf.category_types = ['Table', 'StorageFurniture']
else:
    conf.category_types = conf.category_types.split(',')
print(conf.category_types)

if not os.path.exists(conf.tar_data_dir):
    os.makedirs(conf.tar_data_dir)
    print(f'create dir {conf.tar_data_dir}')

datagen = DataGen(conf.num_processes)

with open(os.path.join(conf.src_data_dir, 'data_tuple_list.txt'), 'r') as fin:
    for l in fin.readlines():
        record_id = l.rstrip()
        datagen.add_one_collect_compare_job(conf.src_data_dir,
                                      record_id,
                                      conf.tar_data_dir)

print("start collect")
datagen.start_all()

data_tuple_list = datagen.join_all()

with open(os.path.join(conf.tar_data_dir, conf.out_fn), 'w') as fout:
    for item in data_tuple_list:
        fout.write(item.split('/')[-1] + '\n')

