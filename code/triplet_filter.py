import os
import shutil
import json

global data_tuple_list
data_tuple_list = []


dir = "../data/gt_pushing_train_cpbs"
new_dir = "../data/gt_pushing_train_cpsm"
newnew_dir = "../data/gt_pushing_train_cpct"
print(os.path.exists(dir))

with open(os.path.join(dir, 'data_tuple_list.txt'), 'r') as fout:
    for root, dirs, files in os.walk(dir):
        for name in dirs:
            try:
                cur_shape_id, cur_category, occlusion_shape_id, occlusion_category, cur_epoch_id, cur_primact_type, cur_trial_id = name.split('_')
            except:
                continue
            new_trial_id = str(int(cur_trial_id))
            name = '_'.join([cur_shape_id, cur_category, occlusion_shape_id, occlusion_category, cur_epoch_id, cur_primact_type, cur_trial_id])
            new_name = '_'.join(
                [cur_shape_id, cur_category, occlusion_shape_id, occlusion_category, cur_epoch_id, cur_primact_type,
                 new_trial_id])
            newnew_trial_id = str(int(cur_trial_id))
            newnew_name = '_'.join(
                [cur_shape_id, cur_category, occlusion_shape_id, occlusion_category, cur_epoch_id, cur_primact_type,
                 newnew_trial_id])

            print(name)
            file_path = os.path.join(dir, name)
            resultfile = os.path.join(file_path, 'result.json')
            print(resultfile)
            if os.path.exists(resultfile) and os.path.exists(os.path.join(new_dir, new_name, 'result.json')) and os.path.exists(os.path.join(newnew_dir, newnew_name, 'result.json')):

                data_tuple_list.append(name)
            else:
                # shutil.rmtree(file_path, ignore_errors=True) # Other folders may be deleted such as succ_imgs & fail_imgs
                pass

with open(os.path.join(dir,'filter_tuple_list.txt'), 'w') as fout:
    for item in data_tuple_list:
        fout.write(item.split('/')[-1]  + '\n')