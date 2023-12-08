python gen_offline_data.py \
  --data_dir ../data/gt_pushing_train_cpbs \
  --data_fn ../stats/train_1cats_train_data_list.txt \
  --primact_types pushing \
  --num_processes 25 \
  --num_epochs 500 \
  --category_types StorageFurniture,Table \
  --data_split train_cat_train_shape

python gen_offline_data.py   \
--data_dir ../data/gt_pushing_test  \
 --data_fn ../stats/train_1cats_train_data_list.txt  \
  --primact_types pushing   \
  --num_processes 25   \
  --num_epochs 50   \
  --category_types StorageFurniture,Table   \
  --data_split test_cat

python gen_similar_data.py \
  --src_data_dir ../data/gt_pushing_train_cpbs \
  --tar_data_dir ../data/gt_pushing_train_cpsm \
  --data_fn ../stats/train_1cats_train_data_list.txt \
  --primact_types pushing \
  --num_processes 25 \
  --num_epochs 500 \
  --out_fn data_tuple_list.txt

python gen_contrastive_data.py \
  --src_data_dir ../data/gt_pushing_train_cpbs \
  --tar_data_dir ../data/gt_pushing_train_cpct \
  --data_fn ../stats/train_1cats_train_data_list.txt \
  --primact_types pushing \
  --num_processes 25 \
  --num_epochs 500 \
  --out_fn data_tuple_list.txt

python triplet_filter.py