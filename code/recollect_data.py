import os
import sys
import shutil
import numpy as np
from PIL import Image
from utils import get_global_position_from_camera, save_h5, get_random_number, export_pts, render_pts_label_png
import cv2
import json
from argparse import ArgumentParser

import imageio
from sapien.core import Pose
from env import Env, ContactError
from camera import Camera
from robots.panda_robot import Robot

from subprocess import call

parser = ArgumentParser()
parser.add_argument('src_data_dir', type=str)
parser.add_argument('record_name', type=str)
parser.add_argument('tar_data_dir', type=str)
parser.add_argument('--random', action='store_true', default=False)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--fix_targetp', action='store_true', default=False, help='fix_target [default: False]')
parser.add_argument('--add_occlusion_num', type=int, default=0, help='add_occlusion_num [default: 0]')
parser.add_argument('--rm_occlusion', type=str, default='none', help='rm_occlusion [default: None]')
parser.add_argument('--data_split', type=str, default='train_cat_train_shape')
args = parser.parse_args()


shape_id, category, occlusion_shape_id, occlusion_category, cnt_id, primact_type, trial_id = args.record_name.split('_')
cnt_id = int(cnt_id)
trial_id = int(trial_id)
if args.fix_targetp and args.rm_occlusion == 'min':
    # trial_id = (int(trial_id) + 1000)
    out_dir = os.path.join(args.tar_data_dir, '%s_%s_%s_%s_%d_%s_%d' % (shape_id, category, occlusion_shape_id, occlusion_category, cnt_id, primact_type, trial_id))
elif args.fix_targetp and args.add_occlusion_num > 0:
    # trial_id = (int(trial_id) + 2000)
    out_dir = os.path.join(args.tar_data_dir, '%s_%s_%s_%s_%d_%s_%d' % (shape_id, category, occlusion_shape_id, occlusion_category, cnt_id, primact_type, trial_id))
else:
    # trial_id = (int(trial_id)+3000)
    out_dir = os.path.join(args.tar_data_dir, '%s_%s_%s_%s_%d_%s_%d' % (shape_id, category, occlusion_shape_id, occlusion_category, cnt_id, primact_type, trial_id))

succ_img_dir = os.path.join(args.tar_data_dir, 'succ_imgs')
if not os.path.exists(succ_img_dir):
    os.mkdir(succ_img_dir)
fail_img_dir = os.path.join(args.tar_data_dir, 'fail_imgs')
if not os.path.exists(fail_img_dir):
    os.mkdir(fail_img_dir)


all_tar_dir = os.path.join(args.tar_data_dir, 'tar_imgs')
if not os.path.exists(all_tar_dir):
    os.mkdir(all_tar_dir)

all_reach_dir = os.path.join(args.tar_data_dir, 'reach_imgs')
if not os.path.exists(all_reach_dir):
    os.mkdir(all_reach_dir)


if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)
flog = open(os.path.join(out_dir, 'log.txt'), 'w')
out_info = dict()


# load old-data result.json
with open(os.path.join(args.src_data_dir, args.record_name, 'result.json'), 'r') as fin:
    replay_data = json.load(fin)
# set random seed
if not args.random:
    try:
        np.random.seed(replay_data['random_seed'])
        out_info['random_seed'] = replay_data['random_seed']
    except:
        rd_seed = np.random.randint(10000000)
        np.random.seed(rd_seed)
        out_info['random_seed'] = rd_seed

# setup env
env = Env(flog=flog, show_gui=(not args.no_gui))

# setup camera
cam = Camera(env, dist=5.0, phi=np.pi/10, theta=np.pi, fixed_position=True)
out_info['camera_metadata'] = cam.get_metadata_json()
if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi+cam.theta, -cam.phi)

# load shape
object_urdf_fn = '../data/sapien_dataset/%s/mobility.urdf' % shape_id
flog.write('object_urdf_fn: %s\n' % object_urdf_fn)
object_material = env.get_material(4, 4, 0.01)
state = replay_data['object_state']
flog.write('Object State: %s\n' % state)
out_info['object_state'] = state

env.load_object(object_urdf_fn, object_material, state=state)
try:
    joint_angles = replay_data['object_joint_angles']
except KeyError:
    joint_angles = replay_data['joint_angles']
env.set_object_joint_angles(joint_angles)
out_info['joint_angles'] = joint_angles
out_info['joint_angles_lower'] = env.joint_angles_lower
out_info['joint_angles_upper'] = env.joint_angles_upper
cur_qpos = env.get_object_qpos()

env.step()
env.render()

rgb, depth = cam.get_observation()
marked_rgb = (rgb*255).astype(np.uint8)
Image.fromarray(marked_rgb).save(os.path.join(out_dir, 'obj.png'))

cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])

# Save masked points
mask = (cam_XYZA[:, :, 3] > 0.5)
pc = cam_XYZA[mask, :3]
export_pts(os.path.join(out_dir, 'obj.pts'), pc)


robot_m_x = replay_data["robot_init_x"]
robot_m_y = replay_data["robot_init_y"]
robot_m_z = 0.0

MIN_OCC_DIS = []
# load occlusion
for iiiii in range(replay_data['occlusion_num']-1):
    occlusion_urdf_fn = replay_data['occlusion_type'+str(iiiii)]
    flog.write('occlusion_urdf_fn: %s\n' % occlusion_urdf_fn)
    occlusion_material = env.get_material(4, 4, 0.01)
    init_scale = replay_data['occlusion_init_scale'+str(iiiii)]
    if init_scale <= 0:
        continue
    try:
        init_z = replay_data['occlusion_init_z'+str(iiiii)]
        init_x = replay_data['occlusion_init_x'+str(iiiii)]
        init_y = replay_data['occlusion_init_y'+str(iiiii)]
    except:
        replay_data['occlusion_num'] += 1
    # if iiiii == rm_id:
    #     continue
    env.load_occlusion(occlusion_urdf_fn,
                       {'init_z': init_z, 'init_x': init_x, 'init_y': init_y, 'init_scale': init_scale},
                       occlusion_material, state='closed')

    objv, objf = env.get_global_mesh(env.scene_objects[-1])
    obj_fullpc = env.sample_pc(objv, objf, n_points=1000)
    obj_xyz1 = np.c_[obj_fullpc, np.ones(obj_fullpc.shape[0])]
    pc_obj = obj_xyz1.T[:3, :].T

    min_dis = np.min(np.linalg.norm(pc_obj - np.array([robot_m_x, robot_m_y, robot_m_z]), axis=1))
    MIN_OCC_DIS.append(min_dis)
    out_info['occlusion_type' + str(iiiii)] = occlusion_urdf_fn

    out_info['occlusion_init_scale' + str(iiiii)] = init_scale
    out_info['occlusion_init_x' + str(iiiii)] = init_x
    out_info['occlusion_init_y' + str(iiiii)] = init_y
    out_info['occlusion_init_z' + str(iiiii)] = init_z


# add new occlusions
with open(os.path.join('../data/sapien_dataset', str(shape_id), 'bounding_box.json'), 'r') as fin:
    bounding_box = json.load(fin)
    box_min = bounding_box['min']
    min_y = box_min[0]
    min_z = box_min[1]
    min_x = box_min[2]
    box_max = bounding_box['max']
    max_y = box_max[0]
    max_z = box_max[1]
    max_x = box_max[2]
#  load cat-freq
cat2freq = dict()
with open('../stats/all_cats_cnt_freq.txt', 'r') as fin:
    for l in fin.readlines():
        cat, _, freq = l.rstrip().split()
        cat2freq[cat] = int(freq)

# load act obj list
cats_train_test_split = 'train' if 'train_cat' in args.data_split else 'test'
with open(os.path.join('../stats', 'act_cats-%s.txt' % cats_train_test_split), 'r') as fin:
    act_cats = [l.rstrip() for l in fin.readlines()]
act_shapes = []
act_shapes_category = []
for act_cat in act_cats:
    with open('../stats/%s-%s.txt' % (act_cat, args.data_split), 'r') as fin:
        for l in fin.readlines():
            act_shape = l.rstrip()
            act_shapes += [act_shape] * cat2freq[act_cat]
            act_shapes_category += [act_cat] * cat2freq[act_cat]


available_min_x = 3*min_x
available_max_x = 3*max_x
available_min_y = 3*min_y
available_max_y = min_y

# smaller_cat = ['Bucket, TrashCan']
big_cat = ['Chair', 'FoldingChair', 'Cart']


for iiiii in range(replay_data['occlusion_num']-1, replay_data['occlusion_num'] - 1 + args.add_occlusion_num):
    idx_choice = np.random.randint(len(act_shapes))
    # occlusion_shape_id = np.random.choice(act_shapes)
    occlusion_shape_id = act_shapes[idx_choice]
    occlusion_cat = act_shapes_category[idx_choice]
    if int(occlusion_shape_id) < 200000:
        occlusion_urdf_fn = '../data/sapien_dataset/%s/mobility.urdf' % occlusion_shape_id
    else:
        occlusion_urdf_fn = '../data/o2o_sapien_dataset/%s/mobility_vhacd.urdf' % occlusion_shape_id
    flog.write('occlusion_urdf_fn: %s\n' % occlusion_urdf_fn)
    occlusion_material = env.get_material(4, 4, 0.01)

    try:
        with open(os.path.join('../data/sapien_dataset', str(occlusion_shape_id), 'bounding_box.json'), 'r') as fin:
            occlu_bounding_box = json.load(fin)

            occlu_box_min = occlu_bounding_box['min']
            occlu_box_max = occlu_bounding_box['max']

            occlu_min_y = occlu_box_min[0]
            occlu_min_z = occlu_box_min[1]
            occlu_min_x = occlu_box_min[2]
            occlu_max_y = occlu_box_max[0]
            occlu_max_z = occlu_box_max[1]
            occlu_max_x = occlu_box_max[2]
    except:
        occlu_min_x, occlu_min_y, occlu_min_z, occlu_max_x, occlu_max_y, occlu_max_z = env.get_urdf_bounding_box(
            occlusion_urdf_fn)

    init_scale = 0.35
    if occlusion_cat in big_cat:
        init_scale = get_random_number(0.5, 0.58)
    # init_scale = 1

    init_z = float(env.ground_z) - occlu_min_z * init_scale

    init_x = get_random_number(1.4 * (min_x - occlu_max_x * init_scale), 1.01 * (min_x - occlu_max_x * init_scale))

    if np.random.rand() < 0.5:
        init_y = get_random_number(1.45 * min_y, 1.25 * min_y)
    else:
        init_y = get_random_number(1.25 * max_y, 1.45 * max_y)

    env.load_occlusion(occlusion_urdf_fn,
                       {'init_z': init_z, 'init_x': init_x, 'init_y': init_y, 'init_scale': init_scale},
                       occlusion_material, state='closed')
    out_info['occlusion_type' + str(iiiii)] = occlusion_urdf_fn

    out_info['occlusion_init_scale' + str(iiiii)] = init_scale
    out_info['occlusion_init_x' + str(iiiii)] = init_x
    out_info['occlusion_init_y' + str(iiiii)] = init_y
    out_info['occlusion_init_z' + str(iiiii)] = init_z

out_info['occlusion_num'] = env.scene_objects.__len__()
flog.write('occlusion_num %s\n'%env.scene_objects.__len__())

if args.rm_occlusion != 'none' and len(MIN_OCC_DIS) != 0:
    if args.rm_occlusion == 'random':
        rm_id = np.random.randint(len(MIN_OCC_DIS))
    elif args.rm_occlusion == 'min':
        rm_id = np.argmin(MIN_OCC_DIS)
    env.scene.remove_articulation(env.scene_objects[1+rm_id])
    env.scene_objects.remove(env.scene_objects[1+rm_id])
    out_info['rm_id'] = int(rm_id)
    flog.write('Remove Occlusion %s\n'%rm_id)
    print('Remove occlusion!%s\n'%rm_id)

cur_qpos = env.get_scene_qpos()

# simulate some steps for the object to stay rest
still_timesteps = 0
wait_timesteps = 0
while still_timesteps < 5000 and wait_timesteps < 20000:
    env.step()
    env.render()
    cur_new_qpos = env.get_scene_qpos()
    if np.max(np.abs(np.array(cur_new_qpos) - np.array(cur_qpos))) < 1e-6: #and (not invalid_contact):
        still_timesteps += 1
    else:
        still_timesteps = 0
    cur_qpos = cur_new_qpos
    wait_timesteps += 1


### use the GT vision
rgb, depth = cam.get_observation()
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(out_dir, 'rgb.png'))

cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
save_h5(os.path.join(out_dir, 'cam_XYZA.h5'), \
        [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'), \
         (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'), \
         (cam_XYZA_pts.astype(np.float32), 'pc', 'float32'), \
        ])

gt_nor = cam.get_normal_map()
Image.fromarray(((gt_nor+1)/2*255).astype(np.uint8)).save(os.path.join(out_dir, 'gt_nor.png'))

# Save applicable mask
object_movable_link_ids = env.movable_link_ids
gt_applicable_link_mask = cam.get_movable_link_mask(object_movable_link_ids)
Image.fromarray((gt_applicable_link_mask>0).astype(np.uint8)*255).save(os.path.join(out_dir, 'applicable_mask.png'))

# Save interaction mask
target_link_ids = [replay_data["target_object_part_actor_id"]]
# target_link_ids = random.choices(object_movable_link_ids)
gt_movable_link_mask = cam.get_movable_link_mask(target_link_ids)
# Image.fromarray((gt_movable_link_mask>0).astype(np.uint8)*255).save(os.path.join(out_dir, 'interaction_mask.png'))
# Save applicable mask
gt_handle_mask = cam.get_handle_mask()
gt_handle_mask = gt_handle_mask * gt_applicable_link_mask
Image.fromarray((gt_handle_mask>0).astype(np.uint8)*255).save(os.path.join(out_dir, 'handle_mask.png'))

# Save object mask
object_full_link_ids = []
for l in env.object.get_links():
    object_full_link_ids.append(l.get_id())
gt_object_mask = cam.get_movable_link_mask(object_full_link_ids)
Image.fromarray((gt_object_mask>0).astype(np.uint8)*255).save(os.path.join(out_dir, 'object_mask.png'))


# Save masked points
mask = (cam_XYZA[:, :, 3] > 0.5)
pc = cam_XYZA[mask, :3]
export_pts(os.path.join(out_dir, 'scene_pc_cam.pts'), pc)

Pull_Handle = False
# sample a pixel to interact
if args.fix_targetp:
    x, y = replay_data['pixel_locs'][0], replay_data['pixel_locs'][1]
    if gt_movable_link_mask[x, y] == 0:
        flog.write('ERROR: <x: %d, y: %d> not in the gt_movable_link_mask! Quit!' % (x, y))
        exit(1)
else:
    # applicable mask
    left_link_ids = list(set(object_movable_link_ids) - set(target_link_ids))
    print('left_link_ids: ', left_link_ids)
    print('target_link_ids: ', target_link_ids)
    print('object_movable_link_ids: ', object_movable_link_ids)
    gt_left_link_mask = cam.get_movable_link_mask(left_link_ids)


    xs, ys = np.where(gt_left_link_mask > 0)
    if len(xs) == 0:
        print('No Movable Pixel! Quit!\n')
        flog.write('No Movable Pixel! Quit!\n')
        flog.close()
        env.close()
        exit(1)
    idx = np.random.randint(len(xs))  # Randomly sample a pixel to interact
    x, y = xs[idx], ys[idx]

if gt_handle_mask[x, y] > 0:
    Pull_Handle = True

out_info['pixel_locs'] = [int(x), int(y)]
# Save interaction mask
target_link_id = object_movable_link_ids[gt_applicable_link_mask[x, y]-1]

env.set_target_object_part_actor_id(target_link_id)
out_info['target_object_part_actor_id'] = env.target_object_part_actor_id
out_info['target_object_part_joint_id'] = env.target_object_part_joint_id

gt_movable_link_mask = cam.get_movable_link_mask([target_link_id])
Image.fromarray((gt_movable_link_mask>0).astype(np.uint8)*255).save(os.path.join(out_dir, 'interaction_mask.png'))


# get pixel 3D pulling direction (cam/world)
direction_cam = gt_nor[x, y, :3]
direction_cam /= np.linalg.norm(direction_cam)
out_info['direction_camera'] = direction_cam.tolist()
flog.write('Direction Camera: %f %f %f\n' % (direction_cam[0], direction_cam[1], direction_cam[2]))
direction_world = cam.get_metadata()['mat44'][:3, :3] @ direction_cam
out_info['direction_world'] = direction_world.tolist()
flog.write('Direction World: %f %f %f\n' % (direction_world[0], direction_world[1], direction_world[2]))
flog.write('mat44: %s\n' % str(cam.get_metadata()['mat44']))

action_direction_cam = direction_cam # Just in the normal direction
if action_direction_cam @ direction_cam > 0:
    action_direction_cam = -action_direction_cam
out_info['gripper_direction_camera'] = action_direction_cam.tolist()
action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ action_direction_cam
out_info['gripper_direction_world'] = action_direction_world.tolist()

# get pixel 3D position (cam/world)
position_cam = cam_XYZA[x, y, :3]
out_info['position_cam'] = position_cam.tolist()
position_cam_xyz1 = np.ones((4), dtype=np.float32)
position_cam_xyz1[:3] = position_cam
position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
position_world = position_world_xyz1[:3]
out_info['position_world'] = position_world.tolist()

### setup robot
robot_urdf_fn = "../assets/robot/panda/panda.urdf"
robot_material = env.get_material(4, 4, 0.01)

robot_init_x = replay_data["robot_init_x"]
robot_init_y = replay_data["robot_init_y"]
robot_init_z = float(env.ground_z)
robot_init_scale = 1

robot = Robot(env, robot_urdf_fn, robot_init_scale, robot_material, open_gripper=True)#('pulling' in primact_type))

start_pose = Pose([robot_init_x, robot_init_y, robot_init_z], [1, 0, 0, 0])
robot.robot.set_root_pose(start_pose)
robot_mat44 = start_pose.to_transformation_matrix()



for idx, obj in enumerate(env.scene_objects):
    objv, objf =env.get_global_mesh(obj)
    obj_fullpc = env.sample_pc(objv, objf)
    obj_xyz1 = np.c_[obj_fullpc, np.ones(obj_fullpc.shape[0])]
    obj_robot_xyz1T = np.linalg.inv(robot_mat44) @ obj_xyz1.T
    obj_robot_xyz = obj_robot_xyz1T[:3, :].T
    robot.add_point_cloud(obj_robot_xyz)

out_info['robot_type'] = robot_urdf_fn
out_info['robot_init_scale'] = robot_init_scale
out_info['robot_init_x'] = robot_init_x
out_info['robot_init_y'] = robot_init_y
out_info['robot_init_z'] = robot_init_z
print('robot init pose is'+str([robot_init_x, robot_init_y, robot_init_z]))
env.step()
env.render()

cam.get_observation()

init_pose = robot.end_effector.get_pose()
init_rotmat = init_pose.to_transformation_matrix()
out_info['init_rotmat_world'] = init_rotmat.tolist()

rgb, depth = cam.get_observation()
marked_rgb = (rgb*255).astype(np.uint8)
marked_rgb = cv2.circle(marked_rgb, (y, x), radius=3, color=(0, 0, 255), thickness=5)
Image.fromarray(marked_rgb).save(os.path.join(out_dir, 'point_to_interact.png'))


if replay_data['result'] == 'VALID' and args.fix_targetp:
    out_info['result'] = 'VALID'
    # save results
    with open(os.path.join(out_dir, 'result.json'), 'w') as fout:
        print(f'save to {out_dir}')
        json.dump(out_info, fout)
    flog.write('result: VALID\n')
    # close the file
    flog.close()
    env.close()
    exit(0)

if args.add_occlusion_num > 0 and args.rm_occlusion == 'none' and args.fix_targetp:
    out_info['result'] = replay_data['result']
    # save results
    with open(os.path.join(out_dir, 'result.json'), 'w') as fout:
        print(f'save to {out_dir}')
        json.dump(out_info, fout)
    flog.write('result: same\n')
    # close the file
    flog.close()
    env.close()
    exit(0)


# compute final pose
up = np.array(action_direction_world, dtype=np.float32)
# forward = np.random.randn(3).astype(np.float32)
forward = init_rotmat[:3, 0]
while abs(up @ forward) > 0.99:
    forward = np.random.randn(3).astype(np.float32)
left = np.cross(up, forward)
left /= np.linalg.norm(left)
forward = np.cross(left, up)
forward /= np.linalg.norm(forward)
out_info['gripper_forward_direction_world'] = forward.tolist()
forward_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ forward
out_info['gripper_forward_direction_camera'] = forward_cam.tolist()
rotmat = np.eye(4).astype(np.float32)
rotmat[:3, 0] = forward
rotmat[:3, 1] = left
rotmat[:3, 2] = up

final_dist = 0.1
if primact_type == 'pushing-left' or primact_type == 'pushing-up':
    final_dist = 0.11

final_rotmat = np.array(rotmat, dtype=np.float32)
final_rotmat[:3, 3] = position_world - action_direction_world * final_dist  # Minus half-length of gripper
robot_base_final_rotmat = np.linalg.inv(robot_mat44) @ final_rotmat
robot_base_final_pose = Pose().from_transformation_matrix(robot_base_final_rotmat)
out_info['target_rotmat_world'] = final_rotmat.tolist()

start_rotmat = np.array(rotmat, dtype=np.float32)
start_rotmat[:3, 3] = position_world - action_direction_world * 0.15
robot_base_start_rotmat = np.linalg.inv(robot_mat44) @ start_rotmat
robot_base_start_pose = Pose().from_transformation_matrix(robot_base_start_rotmat)
out_info['start_rotmat_world'] = start_rotmat.tolist()

robot.close_gripper()
env.render()

rgb, depth = cam.get_observation()
Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(out_dir, 'rgb_withrobot.png'))

rgb, depth = cam.get_observation()
marked_rgb = (rgb*255).astype(np.uint8)
marked_rgb = cv2.circle(marked_rgb, (y, x), radius=3, color=(0, 0, 255), thickness=5)
Image.fromarray(marked_rgb).save(os.path.join(all_tar_dir, '%s_%s_%s_%s_%d_%s_%d.png' % (
    shape_id, category, occlusion_shape_id, occlusion_category, cnt_id, primact_type,
    trial_id)))

start_pose_list = np.concatenate((robot_base_start_pose.p, robot_base_start_pose.q)).tolist()
reached, reachimgs = robot.move_to_pose(start_pose_list, use_point_cloud=True, use_attach=False, vis_gif=True, vis_gif_interval=100,
                            cam=cam)


rgb, depth = cam.get_observation()
marked_rgb = (rgb*255).astype(np.uint8)
marked_rgb = cv2.circle(marked_rgb, (y, x), radius=3, color=(0, 0, 255), thickness=5)
Image.fromarray(marked_rgb).save(os.path.join(all_reach_dir, '%s_%s_%s_%s_%d_%s_%d.png' % (
    shape_id, category, occlusion_shape_id, occlusion_category, cnt_id, primact_type,
    trial_id)))

if reached != 0:
    out_info['result'] = 'REACH_IKFailure'
    with open(os.path.join(out_dir, 'result.json'), 'w') as fout:
        print(f'save to {out_dir}')
        json.dump(out_info, fout)

    # imageio.mimsave(os.path.join(fail_img_dir, '%s_%s_%s_%s_%d_%s_%d.gif' % (
    # shape_id, category, occlusion_shape_id, occlusion_category, cnt_id, primact_type, trial_id)),
    #                 reachimgs)
    # close the file
    flog.close()
    env.close()
    exit(0)
print('Move. Now wait.')
robot.wait_n_steps(1000)

action_direction = None
if 'left' in primact_type:
    action_direction = forward
elif 'up' in primact_type:
    action_direction = left

if action_direction is not None:
    end_rotmat = np.array(rotmat, dtype=np.float32)
    end_rotmat[:3, 3] = position_world - action_direction_world * final_dist + action_direction * 0.05
    robot_base_end_rotmat = np.linalg.inv(robot_mat44) @ end_rotmat
    robot_base_end_pose = Pose().from_transformation_matrix(robot_base_end_rotmat)
    out_info['end_rotmat_world'] = end_rotmat.tolist()

init_target_part_qpos = env.get_target_part_qpos()
# activate contact checking
env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, robot.arm_actor_ids, 'pushing' in primact_type)

if not args.no_gui:
    ### wait to start
    env.wait_to_start()

### main steps
out_info['start_target_part_qpos'] = env.get_target_part_qpos()

target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1


success = False
success_grasp = False
try:
    if 'pushing' in primact_type:
        robot.close_gripper()
    elif 'pulling' in primact_type:
        robot.open_gripper()
    apprchimgs = robot.move_to_target_pose(final_rotmat, 80, cam=cam, vis_gif=True, vis_gif_interval=100, visu=False)

    env.render()
    rgb_final_pose, _ = cam.get_observation()
    Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'reach_target_pose.png'))
    if not args.no_gui:
        env.wait_to_start()

    imgswait = robot.wait_n_steps(2000, cam=cam, vis_gif=True, vis_gif_interval=100, visu=False)
    # succ_images.extend(imgs)
    if not args.no_gui:
        env.wait_to_start()
    if 'pulling' in primact_type:
        robot.close_gripper()
        robot.wait_n_steps(500)
        now_qpos = robot.robot.get_qpos().tolist()
        finger1_qpos = now_qpos[-1]
        finger2_qpos = now_qpos[-2]
        if finger1_qpos + finger2_qpos > 0.01:
            success_grasp = True

    if 'left' in primact_type or 'up' in primact_type:
        robot.move_to_target_pose(end_rotmat, 80, cam=cam, vis_gif=True, vis_gif_interval=100, visu=False)
        robot.wait_n_steps(2000)

    if primact_type == 'pulling':
        pullimgs = robot.move_to_target_pose(start_rotmat, 80, cam=cam, vis_gif=True, vis_gif_interval=100, visu=False)
        imgswaitaft = robot.wait_n_steps(2000, cam=cam, vis_gif=True, vis_gif_interval=100, visu=False)


    now_qpos = robot.robot.get_qpos().tolist()
    now_gripper_root_position = position_world - 0.15 * action_direction_world + now_qpos[0] * forward + now_qpos[1] * left + now_qpos[2] * up

    final_target_part_qpos = env.get_target_part_qpos()
    distance = np.abs((init_target_part_qpos) - (final_target_part_qpos))
    print('initial target part pose is %s' % str(np.abs((init_target_part_qpos))))
    print('final target part pose is %s' % str(np.abs((final_target_part_qpos))))

    target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
    position_world_xyz1_end = target_link_mat44 @ position_local_xyz1
    flog.write('touch_position_world_xyz_start: %s\n' % str(position_world_xyz1))
    flog.write('touch_position_world_xyz_end: %s\n' % str(position_world_xyz1_end))
    out_info['touch_position_world_xyz_start'] = position_world_xyz1[:3].tolist()
    out_info['touch_position_world_xyz_end'] = position_world_xyz1_end[:3].tolist()

    # TASK SUCCESS GOAL
    abs_thres = 0.01
    rel_thres = 0.5
    dp_thres = 0.5
    abs_motion = abs(final_target_part_qpos - init_target_part_qpos)
    j = env.target_object_part_joint_id
    tot_motion = env.joint_angles_upper[j] - env.joint_angles_lower[j] + 1e-8
    success = (abs_motion > abs_thres) or (abs_motion / tot_motion > rel_thres)
    if not success:
        pass
    elif primact_type == 'pulling':
        mov_dir = np.array(position_world_xyz1_end[:3], dtype=np.float32) - \
                  np.array(position_world_xyz1[:3], dtype=np.float32)
        mov_dir /= np.linalg.norm(mov_dir)
        intended_dir = -np.array(action_direction_world, dtype=np.float32)
        success = (intended_dir @ mov_dir > dp_thres)
        if not success_grasp:
            success = False
    if Pull_Handle:
        success = True

    save_imgs = reachimgs+apprchimgs+imgswait
    if primact_type == 'pulling':
        save_imgs += pullimgs+imgswaitaft
    if success == True:
        imageio.mimsave(os.path.join(succ_img_dir, '%s_%s_%s_%s_%d_%s_%d.gif' % (
            shape_id, category, occlusion_shape_id, occlusion_category, cnt_id, primact_type, trial_id)),
                        save_imgs)
    else:
        imageio.mimsave(os.path.join(fail_img_dir, '%s_%s_%s_%s_%d_%s_%d.gif' % (
            shape_id, category, occlusion_shape_id, occlusion_category, cnt_id, primact_type, trial_id)),
                        save_imgs)
    flog.write('Target pose change distance: %s\n' % str(distance))

except ContactError:

    imageio.mimsave(os.path.join(fail_img_dir, '%s_%s_%s_%s_%d_%s_%d.gif' % (
        shape_id, category, occlusion_shape_id, occlusion_category, cnt_id, primact_type, trial_id)),
                    reachimgs)


if success:
    out_info['result'] = 'VALID'
    out_info['final_target_part_qpos'] = env.get_target_part_qpos()
else:
    out_info['result'] = 'FAILURE'

# save results
with open(os.path.join(out_dir, 'result.json'), 'w') as fout:
    print(f'save to {out_dir}')
    json.dump(out_info, fout)

# close the file
flog.close()

if args.no_gui:
    # close env
    env.close()
else:
    if success:
        print('[Successful Interaction] Done. Ctrl-C to quit.')
        ### wait forever
        robot.wait_n_steps(100000000000)
    else:
        print('[Unsuccessful Interaction] Quit.')
        # close env
        env.close()

