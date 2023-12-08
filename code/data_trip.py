import os
import h5py
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import json
from progressbar import ProgressBar
from pyquaternion import Quaternion
from camera import Camera
from geometry_utils import load_pts


class SAPIENVisionDatasetTrip(data.Dataset):

    def __init__(self, category_types, data_features, \
            env_name=None, buffer_max_num=None, img_size=224, \
            no_true_false_equal=False, no_aug_neg_data=False, only_true_data=False, contrastive_dir_p=None, contrastive_dir_n=None):
        self.category_types = category_types

        self.env_name = env_name
        self.buffer_max_num = buffer_max_num
        self.img_size = img_size

        self.no_true_false_equal = no_true_false_equal
        self.no_aug_neg_data = no_aug_neg_data
        self.only_true_data = only_true_data

        # data buffer
        self.true_data = []
        self.false_data = []

        # data buffer
        self.true_data_p = []
        self.false_data_p = []
        self.true_data_n = []
        self.false_data_n = []

        # data features
        self.data_features = data_features
        self.contrastive_dir_p = contrastive_dir_p
        self.contrastive_dir_n = contrastive_dir_n

    def load_data(self, data_list):
        bar = ProgressBar()
        for i in bar(range(len(data_list))):
            cur_dir = data_list[i]
            cur_shape_id, cur_category, occlusion_shape_id, occlusion_category, cur_epoch_id, cur_primact_type, cur_trial_id  = cur_dir.split('/')[-1].split('_')
            with open(os.path.join(cur_dir, 'result.json'), 'r') as fin:
                result_data = json.load(fin)

                ori_pixel_ids = np.array(result_data['pixel_locs'], dtype=np.int32)
                pixel_ids = np.round(np.array(result_data['pixel_locs'], dtype=np.float32) / 448 * self.img_size).astype(np.int32)

                # success = result_data['result']
                base_success = self.check_success(result_data, cur_primact_type)

                robot_init_x = np.float32(result_data['robot_init_x'])
                robot_init_y = np.float32(result_data['robot_init_y'])
                robot_init_z = np.float32(result_data['robot_init_z'])

                robot_p = np.array([robot_init_x, robot_init_y, robot_init_z], dtype=np.float32)

                cam2cambase = np.array(result_data['camera_metadata']['cam2cambase'], dtype=np.float32)

                # load original data
                if base_success:
                    cur_data = (cur_dir, cur_shape_id, cur_category, cur_epoch_id, cur_trial_id, \
                            ori_pixel_ids, pixel_ids, True, True, cam2cambase, robot_p)
                    self.true_data.append(cur_data)
                else:
                    if not self.only_true_data:
                        cur_data = (cur_dir, cur_shape_id, cur_category, cur_epoch_id, cur_trial_id, \
                                ori_pixel_ids, pixel_ids, True, False, cam2cambase, robot_p)
                        self.false_data.append(cur_data)
            if self.contrastive_dir_p is None or self.contrastive_dir_n is None:
                print("continue")
                continue

            if int(cur_trial_id) < 1000:
                contrastive_p_trial_id = int(cur_trial_id)
                contrastive_dir_p = self.contrastive_dir_p
            else:
                contrastive_p_trial_id = int(cur_trial_id) - 2000
                contrastive_dir_p = self.contrastive_dir_n
            with open(os.path.join(contrastive_dir_p, cur_shape_id+'_'+ cur_category+'_'+ occlusion_shape_id+'_'+ occlusion_category+'_'+ cur_epoch_id+'_'+ cur_primact_type+'_'+  str(contrastive_p_trial_id), 'result.json'), 'r') as fin:
                result_data = json.load(fin)

                ori_pixel_ids = np.array(result_data['pixel_locs'], dtype=np.int32)

                # print("pos:", ori_pixel_ids)
                pixel_ids = np.round(
                    np.array(result_data['pixel_locs'], dtype=np.float32) / 448 * self.img_size).astype(np.int32)

                success = result_data['result']
                success = self.check_success(result_data, cur_primact_type)

                robot_init_x = np.float32(result_data['robot_init_x'])
                robot_init_y = np.float32(result_data['robot_init_y'])
                robot_init_z = np.float32(result_data['robot_init_z'])

                robot_p = np.array([robot_init_x, robot_init_y, robot_init_z], dtype=np.float32)

                cam2cambase = np.array(result_data['camera_metadata']['cam2cambase'], dtype=np.float32)

                # load original data
                if base_success:
                    cur_data = (os.path.join(contrastive_dir_p, cur_shape_id+'_'+ cur_category+'_'+ occlusion_shape_id+'_'+ occlusion_category+'_'+ cur_epoch_id+'_'+ cur_primact_type+'_'+  str(contrastive_p_trial_id)), cur_shape_id, cur_category, cur_epoch_id, cur_trial_id, \
                                ori_pixel_ids, pixel_ids, True, success, cam2cambase, robot_p)
                    self.true_data_p.append(cur_data)
                else:
                    # if not self.only_true_data:
                    cur_data = (os.path.join(contrastive_dir_p, cur_shape_id+'_'+ cur_category+'_'+ occlusion_shape_id+'_'+ occlusion_category+'_'+ cur_epoch_id+'_'+ cur_primact_type+'_'+  str(contrastive_p_trial_id)), cur_shape_id, cur_category, cur_epoch_id, cur_trial_id, \
                                ori_pixel_ids, pixel_ids, True, success, cam2cambase, robot_p)
                    self.false_data_p.append(cur_data)
            if int(cur_trial_id) < 1000:
                contrastive_n_trial_id = int(cur_trial_id)
                contrastive_dir_n = self.contrastive_dir_n
            else:
                contrastive_n_trial_id = int(cur_trial_id) - 3000
                contrastive_dir_n = self.contrastive_dir_p
            with open(os.path.join(contrastive_dir_n, cur_shape_id+'_'+ cur_category+'_'+ occlusion_shape_id+'_'+ occlusion_category+'_'+ cur_epoch_id+'_'+ cur_primact_type+'_'+  str(contrastive_n_trial_id), 'result.json'), 'r') as fin:
                result_data = json.load(fin)

                ori_pixel_ids = np.array(result_data['pixel_locs'], dtype=np.int32)

                # print("neg:", ori_pixel_ids)

                pixel_ids = np.round(
                    np.array(result_data['pixel_locs'], dtype=np.float32) / 448 * self.img_size).astype(np.int32)

                success = result_data['result']
                success = self.check_success(result_data, cur_primact_type)

                robot_init_x = np.float32(result_data['robot_init_x'])
                robot_init_y = np.float32(result_data['robot_init_y'])
                robot_init_z = np.float32(result_data['robot_init_z'])

                robot_p = np.array([robot_init_x, robot_init_y, robot_init_z], dtype=np.float32)

                cam2cambase = np.array(result_data['camera_metadata']['cam2cambase'], dtype=np.float32)

                # load original data
                if base_success:
                    cur_data = (os.path.join(contrastive_dir_n, cur_shape_id+'_'+ cur_category+'_'+ occlusion_shape_id+'_'+ occlusion_category+'_'+ cur_epoch_id+'_'+ cur_primact_type+'_'+  str(contrastive_n_trial_id)), cur_shape_id, cur_category, cur_epoch_id, cur_trial_id, \
                                ori_pixel_ids, pixel_ids, True, success, cam2cambase, robot_p)
                    self.true_data_n.append(cur_data)
                else:
                    # if not self.only_true_data:
                    cur_data = (os.path.join(contrastive_dir_n, cur_shape_id+'_'+ cur_category+'_'+ occlusion_shape_id+'_'+ occlusion_category+'_'+ cur_epoch_id+'_'+ cur_primact_type+'_'+  str(contrastive_n_trial_id)), cur_shape_id, cur_category, cur_epoch_id, cur_trial_id, \
                                ori_pixel_ids, pixel_ids, True, success, cam2cambase, robot_p)
                    self.false_data_n.append(cur_data)
        # delete data if buffer full
        if self.buffer_max_num is not None:
            if len(self.true_data) > self.buffer_max_num:
                self.true_data = self.true_data[-self.buffer_max_num:]
            if len(self.false_data) > self.buffer_max_num:
                self.false_data = self.false_data[-self.buffer_max_num:]

    def check_success(self, result_data, primact_type):
        if result_data['result'] != 'VALID':
            return False
        else:
            return True

    def __len__(self):
        if self.no_true_false_equal:
            return len(self.true_data) + len(self.false_data)
        else:
            return max(len(self.true_data), len(self.false_data)) * 2
            # return min(len(self.true_data), len(self.false_data)) * 2

    def __str__(self):
        strout = '[SAPIENVisionDataset %s %d] img_size: %d, no_aug_neg_data: %s\n' % (self.env_name, len(self), self.img_size, 'True' if self.no_aug_neg_data else 'False')
        strout += '\tTrue: %d False: %d\n' % (len(self.true_data), len(self.false_data))
        strout += 'Total True & False : %d\n' % self.__len__()
        return strout


    def __getitem__(self, index):
        def extract_data_features(self, input_feat):
            cur_dir, shape_id, category, epoch_id, trial_id, ori_pixel_ids, pixel_ids, \
            is_original, result, cam2cambase, robot_p = input_feat

            # pre-load some data
            if any(feat in ['gt_applicable_img', 'gt_applicable_pc'] for feat in self.data_features):
                # HERE!!!
                try:
                    with Image.open(os.path.join(cur_dir, 'applicable_mask.png')) as fimg:
                        gt_applicable_img = np.array(fimg, dtype=np.float32) > 128
                except FileNotFoundError:
                    with Image.open(os.path.join(cur_dir, 'gripper_pc_cambase.png')) as fimg:
                        gt_applicable_img = np.array(fimg, dtype=np.float32) > 128

            if any(feat in ['gt_possible_img', 'gt_possible_pc'] for feat in self.data_features):
                try:
                    with Image.open(os.path.join(cur_dir, 'interaction_mask.png')) as fimg:
                        gt_possible_img = np.array(fimg, dtype=np.float32) > 128
                except FileNotFoundError:
                    with Image.open(os.path.join(cur_dir, 'gripper_pc_cambase.png')) as fimg:
                        gt_possible_img = np.array(fimg, dtype=np.float32) > 128

            if any(feat in ['gt_occlusion_img', 'gt_object_pc'] for feat in self.data_features):
                try:
                    with Image.open(os.path.join(cur_dir, 'object_mask.png')) as fimg:
                        gt_object_img = np.array(fimg, dtype=np.float32) > 128
                except FileNotFoundError:
                    with Image.open(os.path.join(cur_dir, 'gripper_pc_cambase.png')) as fimg:
                        gt_object_img = np.array(fimg, dtype=np.float32) > 128

            if any(feat in ['scene_pc_cam', 'scene_pc_pxids', 'gt_applicable_pc', 'gt_possible_pc', 'gt_object_pc'] for
                   feat in self.data_features):
                x, y = ori_pixel_ids[0], ori_pixel_ids[1]
                with h5py.File(os.path.join(cur_dir, 'cam_XYZA.h5'), 'r') as fin:
                    cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                    cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                    cam_XYZA_pts = fin['pc'][:].astype(np.float32)
                scene_pc_img = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)

                pt = scene_pc_img[x, y, :3]
                if 'scene_pc_pxids' in self.data_features:
                    pc_ptid = np.array([x, y], dtype=np.int32)
                if 'gt_applicable_pc' in self.data_features:
                    gt_applicable_pt = gt_applicable_img[x, y]
                if 'gt_possible_pc' in self.data_features:
                    gt_possible_pt = gt_possible_img[x, y]
                if 'gt_object_pc' in self.data_features:
                    gt_object_pt = gt_object_img[x, y]

                mask = (scene_pc_img[:, :, 3] > 0.5)
                mask[x, y] = False
                scene_pc_cam = scene_pc_img[mask, :3]
                if 'scene_pc_pxids' in self.data_features:
                    grid_x, grid_y = np.meshgrid(np.arange(448), np.arange(448))
                    grid_xy = np.stack([grid_y, grid_x]).astype(np.int32)  # 2 x 448 x 448
                    pc_pxids = grid_xy[:, mask].T
                if 'gt_applicable_pc' in self.data_features:
                    gt_applicable_pc = gt_applicable_img[mask]
                if 'gt_possible_pc' in self.data_features:
                    gt_possible_pc = gt_possible_img[mask]
                if 'gt_object_pc' in self.data_features:
                    gt_object_pc = gt_object_img[mask]

                idx = np.arange(scene_pc_cam.shape[0])
                np.random.shuffle(idx)
                while len(idx) < 30000:
                    idx = np.concatenate([idx, idx])
                idx = idx[:30000 - 1]
                scene_pc_cam = scene_pc_cam[idx, :]
                scene_pc_cam = np.vstack([pt, scene_pc_cam])
                if 'scene_pc_pxids' in self.data_features:
                    pc_pxids = np.vstack([pc_ptid, pc_pxids[idx, :]])
                if 'gt_applicable_pc' in self.data_features:
                    gt_applicable_pc = np.append(gt_applicable_pt, gt_applicable_pc[idx])
                if 'gt_possible_pc' in self.data_features:
                    gt_possible_pc = np.append(gt_possible_pt, gt_possible_pc[idx])
                if 'gt_object_pc' in self.data_features:
                    gt_object_pc = np.append(gt_object_pt, gt_object_pc[idx])

            # output all require features
            data_feats = ()
            for feat in self.data_features:
                if feat == 'rgb':
                    with Image.open(os.path.join(cur_dir, 'rgb.png')) as fimg:
                        out = np.array(fimg.resize((self.img_size, self.img_size)), dtype=np.float32) / 255
                    out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                    data_feats = data_feats + (out,)

                elif feat == 'gt_nor':
                    x, y = ori_pixel_ids[0], ori_pixel_ids[1]
                    with Image.open(os.path.join(cur_dir, 'gt_nor.png')) as fimg:
                        out = np.array(fimg, dtype=np.float32) / 255
                    out = out[x, y, :3] * 2 - 1
                    out = torch.from_numpy(out).unsqueeze(0)
                    data_feats = data_feats + (out,)

                elif feat == 'rgb_start':
                    if is_original:
                        try:
                            with Image.open(os.path.join(cur_dir, 'rgb_withrobot.png')) as fimg:
                                out = np.array(fimg, dtype=np.float32) / 255
                            out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                        except:
                            out = torch.ones(1, 3, 448, 448).float()
                    else:
                        out = torch.ones(1, 3, 448, 448).float()
                    data_feats = data_feats + (out,)

                elif feat == 'rgb_final':
                    if is_original:
                        with Image.open(os.path.join(cur_dir, 'rgb_final.png')) as fimg:
                            out = np.array(fimg, dtype=np.float32) / 255
                        out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                    else:
                        out = torch.ones(1, 3, 448, 448).float()
                    data_feats = data_feats + (out,)

                elif feat == 'rgb_point':
                    if is_original:
                        with Image.open(os.path.join(cur_dir, 'point_to_interact.png')) as fimg:
                            out = np.array(fimg, dtype=np.float32) / 255
                        out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
                    else:
                        out = torch.ones(1, 3, 448, 448).float()
                    data_feats = data_feats + (out,)

                elif feat == 'gt_applicable_img':
                    out = torch.from_numpy(gt_applicable_img).unsqueeze(0)
                    data_feats = data_feats + (out,)

                elif feat == 'gt_applicable_pc':
                    out = torch.from_numpy(gt_applicable_pc).unsqueeze(0)
                    data_feats = data_feats + (out,)

                elif feat == 'gt_object_img':
                    out = torch.from_numpy(gt_object_img).unsqueeze(0)
                    data_feats = data_feats + (out,)

                elif feat == 'gt_object_pc':
                    out = torch.from_numpy(gt_object_pc).unsqueeze(0)
                    data_feats = data_feats + (out,)


                elif feat == 'gt_possible_img':
                    out = torch.from_numpy(gt_possible_img).unsqueeze(0)
                    data_feats = data_feats + (out,)

                elif feat == 'gt_possible_pc':
                    out = torch.from_numpy(gt_possible_pc).unsqueeze(0)
                    data_feats = data_feats + (out,)

                elif feat == 'gt_possible_img':
                    out = torch.from_numpy(gt_possible_img).unsqueeze(0)
                    data_feats = data_feats + (out,)

                elif feat == 'scene_pc_cam':
                    out = torch.from_numpy(scene_pc_cam).unsqueeze(0)
                    data_feats = data_feats + (out,)

                elif feat == 'scene_pc_pxids':
                    out = torch.from_numpy(pc_pxids).unsqueeze(0)
                    data_feats = data_feats + (out,)

                elif feat == 'acting_pc_cambase':
                    pc = load_pts(os.path.join(cur_dir, 'acting_object_cambase.pts'))
                    if self.env_name is not None:
                        pc_z_size = pc[:, 2].max() - pc[:, 2].min()
                        pc[:, 2] += pc_z_size / 2
                        if self.env_name in ['pushing', 'rotating']:
                            pc_x_size = pc[:, 0].max() - pc[:, 0].min()
                            pc[:, 0] -= pc_x_size / 2
                    out = torch.from_numpy(pc).unsqueeze(0)
                    data_feats = data_feats + (out,)

                elif feat == 'robot_pc_cam':
                    pc = load_pts('../assets/robot/panda/panda.pts')

                    out = torch.from_numpy(pc).unsqueeze(0)
                    data_feats = data_feats + (out,)

                elif feat == 'interaction_pc_cambase':
                    try:
                        pc = load_pts(os.path.join(cur_dir, 'interaction_pc_cambase.pts'))
                    except:
                        pc = load_pts(os.path.join(cur_dir, 'interaction_pc_cambase.pts.pts'))
                    idx = np.arange(pc.shape[0])
                    np.random.shuffle(idx)
                    while len(idx) < 10000:
                        idx = np.concatenate([idx, idx])
                    idx = idx[:10000 - 1]
                    pc = pc[idx, :]
                    # print(pc.shape)
                    # if self.env_name is not None:
                    #     pc_z_size = pc[:, 2].max() - pc[:, 2].min()
                    #     pc[:, 2] += pc_z_size / 2
                    #     if self.env_name in ['pushing', 'rotating']:
                    #         pc_x_size = pc[:, 0].max() - pc[:, 0].min()
                    #         pc[:, 0] -= pc_x_size / 2
                    out = torch.from_numpy(pc).unsqueeze(0)
                    data_feats = data_feats + (out,)

                elif feat == 'object_pc_cambase':
                    try:
                        pc = load_pts(os.path.join(cur_dir, 'object_pc_cambase.pts'))
                    except:
                        pc = load_pts(os.path.join(cur_dir, 'object_pc_cambase.pts.pts'))
                    idx = np.arange(pc.shape[0])
                    np.random.shuffle(idx)
                    while len(idx) < 10000:
                        idx = np.concatenate([idx, idx])
                    idx = idx[:10000 - 1]
                    pc = pc[idx, :]
                    # print(pc.shape)
                    # if self.env_name is not None:
                    #     pc_z_size = pc[:, 2].max() - pc[:, 2].min()
                    #     pc[:, 2] += pc_z_size / 2
                    #     if self.env_name in ['pushing', 'rotating']:
                    #         pc_x_size = pc[:, 0].max() - pc[:, 0].min()
                    #         pc[:, 0] -= pc_x_size / 2
                    out = torch.from_numpy(pc).unsqueeze(0)
                    data_feats = data_feats + (out,)


                elif feat == 'cam2cambase':
                    out = np.array(cam2cambase, dtype=np.float32)
                    out = torch.from_numpy(out).unsqueeze(0)
                    data_feats = data_feats + (out,)

                elif feat == 'is_original':
                    data_feats = data_feats + (is_original,)

                elif feat == 'pixel_id':
                    out = torch.from_numpy(pixel_ids).unsqueeze(0)
                    data_feats = data_feats + (out,)

                elif feat == 'result':
                    data_feats = data_feats + (result,)

                elif feat == 'robot_p':
                    data_feats = data_feats + (robot_p,)

                elif feat == 'cur_dir':
                    data_feats = data_feats + (cur_dir,)

                elif feat == 'shape_id':
                    data_feats = data_feats + (shape_id,)

                elif feat == 'occlusion_shape_id':
                    occlusion_shape_id = 6771
                    data_feats = data_feats + (occlusion_shape_id,)

                elif feat == 'category':
                    data_feats = data_feats + (category,)

                elif feat == 'epoch_id':
                    data_feats = data_feats + (epoch_id,)

                elif feat == 'trial_id':
                    data_feats = data_feats + (trial_id,)

                else:
                    raise ValueError('ERROR: unknown feat type %s!' % feat)

            return data_feats

        data_feats_list = []
        if self.no_true_false_equal:
            if index < len(self.false_data):
                cur_dir, shape_id, category, epoch_id, trial_id, ori_pixel_ids, pixel_ids, \
                        is_original, result, cam2cambase, robot_p = \
                            self.false_data[index]
            else:
                cur_dir, shape_id, category, epoch_id, trial_id, ori_pixel_ids, pixel_ids, \
                        is_original, result, cam2cambase, robot_p = \
                            self.true_data[index - len(self.false_data)]
        else:
            if index % 2 == 0 and len(self.false_data_p) != 0 and len(self.false_data_n) != 0:
                data_feats_list.append(extract_data_features(self, self.false_data[(index//2) % len(self.false_data)]))
                data_feats_list.append(extract_data_features(self, self.false_data_p[(index//2) % len(self.false_data_p)]))
                data_feats_list.append(extract_data_features(self, self.false_data_n[(index//2) % len(self.false_data_n)]))
            elif index % 2 == 1 and len(self.true_data_p) != 0 and len(self.true_data_n) != 0:
                data_feats_list.append(extract_data_features(self, self.true_data[(index//2) % len(self.true_data)]))
                data_feats_list.append(extract_data_features(self, self.true_data_p[(index//2) % len(self.true_data_p)]))
                data_feats_list.append(extract_data_features(self, self.true_data_n[(index//2) % len(self.true_data_n)]))
        # print('data_feats_list', len(data_feats_list))
        return data_feats_list
