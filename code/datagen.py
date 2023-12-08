"""
    Batch-generate data
"""

import os
import shutil
import numpy as np
import multiprocessing as mp
from subprocess import call
from utils import printout
import time


class DataGen(object):

    def __init__(self, num_processes, flog=None):
        self.num_processes = num_processes
        self.flog = flog
        
        self.todos = []
        self.processes = []
        self.is_running = False
        self.Q = mp.Queue()

    def __len__(self):
        return len(self.todos)

    def add_one_collect_job(self, data_dir, data_split, shape_id, category, occlusion_shape_id, occlusion_category, cnt_id, primact_type, trial_id, random_seed=-1):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot add a new job while DataGen is running!')
            exit(1)
        if random_seed == -1:
            random_seed = np.random.randint(10000000)
        todo = ('COLLECT', shape_id, category, occlusion_shape_id, occlusion_category, cnt_id, primact_type, data_dir, trial_id, random_seed, data_split)
        self.todos.append(todo)

    def add_one_collect_compare_job(self, src_data_dir, recollect_record_name, tar_data_dir):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot add a new job while DataGen is running!')
            exit(1)

        todo = ('COLLECT_COMPARE', src_data_dir, recollect_record_name, tar_data_dir)
        self.todos.append(todo)
    
    def add_one_recollect_job(self, src_data_dir, recollect_record_name, tar_data_dir):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot add a new job while DataGen is running!')
            exit(1)

        todo = ('RECOLLECT', src_data_dir, recollect_record_name, tar_data_dir)
        self.todos.append(todo)

    def add_one_recollect_pull_job(self, src_data_dir, recollect_record_name, tar_data_dir):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot add a new job while DataGen is running!')
            exit(1)

        todo = ('RECOLLECT_PULL', src_data_dir, recollect_record_name, tar_data_dir)
        self.todos.append(todo)
    
    def add_one_checkcollect_job(self, src_data_dir, dir1, dir2, recollect_record_name, tar_data_dir, x, y):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot add a new job while DataGen is running!')
            exit(1)

        todo = ('CHECKCOLLECT', src_data_dir, recollect_record_name, tar_data_dir, np.random.randint(10000000), x, y, dir1, dir2)
        self.todos.append(todo)
    
    @staticmethod
    def job_func(pid, todos, Q):
        succ_todos = []
        for todo in todos:
            if todo[0] == 'COLLECT':
                cmd = 'xvfb-run -a python collect_data.py %s %s %s %s %d %s --out_dir %s --trial_id %d --random_seed %d --data_split %s --no_gui > /dev/null 2>&1' \
                        % (todo[1], todo[2], todo[3], todo[4], todo[5], todo[6], todo[7], todo[8], todo[9], todo[10])
                folder_name = todo[7]
                print(f'Collect save to {folder_name}')
                job_name = '%s_%s_%s_%s_%d_%s_%s' % (todo[1], todo[2], todo[3], todo[4], todo[5], todo[6], todo[8])
            elif todo[0] == 'COLLECT_COMPARE':
                cmd = 'xvfb-run -a python recollect_data.py %s %s %s --no_gui > /dev/null 2>&1' \
                        % (todo[1], todo[2], todo[3])
                folder_name = todo[3]
                print(f'save to {folder_name}')
                job_name = todo[2]
            elif todo[0] == 'RECOLLECT':
                cmd = 'xvfb-run -a python recollect_data.py %s %s %s --no_gui --fix_targetp --add_occlusion_num 1 > /dev/null 2>&1' \
                        % (todo[1], todo[2], todo[3])
                folder_name = todo[3]
                print(f'Recollect save to {folder_name}')
                job_name = todo[2]
            ret = call(cmd, shell=True)
            print("cmd start")
            if ret == 0:
                print("cmd succ [0]")
                succ_todos.append(os.path.join(folder_name, job_name))
            # elif ret == 2:
            #     succ_todos.append(None)
            else:
                # succ_todos.append(None)
                # shutil.rmtree(os.path.join(folder_name, job_name))
                print("cmd fail [-1]")
                pass
        Q.put(succ_todos)

    def start_all(self):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot start all while DataGen is running!')
            exit(1)

        total_todos = len(self)
        if self.num_processes != 0:
            num_todos_per_process = int(np.ceil(total_todos / self.num_processes))
        else:
            num_todos_per_process = 0
        np.random.shuffle(self.todos)
        for i in range(self.num_processes):
            todos = self.todos[i*num_todos_per_process: min(total_todos, (i+1)*num_todos_per_process)]
            p = mp.Process(target=self.job_func, args=(i, todos, self.Q))
            p.start()
            self.processes.append(p)
        
        self.is_running = True

    def join_all(self):
        if not self.is_running:
            printout(self.flog, 'ERROR: cannot join all while DataGen is idle!')
            exit(1)

        ret = []
        for p in self.processes:
            ret += self.Q.get()

        for p in self.processes:
            p.join()

        self.todos = []
        self.processes = []
        self.Q = mp.Queue()
        self.is_running = False
        return ret


