"""
    Environment with object at center
"""

from __future__ import division
import sapien.core as sapien
from sapien.core import Pose, SceneConfig, OptifuserConfig
from transforms3d.quaternions import axangle2quat, qmult
import numpy as np
from utils import process_angle_limit, get_random_number
import trimesh

class ContactError(Exception):
    pass


class Env(object):
    
    def __init__(self, flog=None, show_gui=True, render_rate=20, timestep=1/500, \
            object_position_offset=0.0, succ_ratio=0.1):
        self.current_step = 0

        self.flog = flog
        self.show_gui = show_gui
        self.render_rate = render_rate
        self.timestep = timestep
        self.succ_ratio = succ_ratio
        self.object_position_offset = object_position_offset
        
        # engine and renderer
        self.engine = sapien.Engine(0, 0.001, 0.005)
        
        render_config = OptifuserConfig()
        render_config.shadow_map_size = 8192
        render_config.shadow_frustum_size = 10
        render_config.use_shadow = False
        render_config.use_ao = True
        
        self.renderer = sapien.OptifuserRenderer(config=render_config)
        self.renderer.enable_global_axes(False)
        
        self.engine.set_renderer(self.renderer)

        # GUI
        self.window = False
        if show_gui:
            self.renderer_controller = sapien.OptifuserController(self.renderer)
            self.renderer_controller.set_camera_position(-3.0+object_position_offset, 1.0, 3.0)
            self.renderer_controller.set_camera_rotation(-0.4, -0.8)

        # scene
        scene_config = SceneConfig()
        scene_config.gravity = [0, 0, -9.81]
        scene_config.solver_iterations = 20
        scene_config.enable_pcm = False
        scene_config.sleep_threshold = 0.0

        self.scene = self.engine.create_scene(config=scene_config)
        if show_gui:
            self.renderer_controller.set_current_scene(self.scene)

        self.scene.set_timestep(timestep)
        # add lights
        # self.scene.set_ambient_light([0.5, 0.5, 0.5])
        # self.scene.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])
        # self.scene.add_point_light([1+object_position_offset, 2, 2], [1, 1, 1])
        # self.scene.add_point_light([1+object_position_offset, -2, 2], [1, 1, 1])
        # self.scene.add_point_light([-1+object_position_offset, 0, 1], [1, 1, 1])
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])
        self.scene.add_point_light([1, 2, 2], [1, 1, 1])
        self.scene.add_point_light([1, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1])


        # default Nones
        self.object = None
        self.object_target_joint = None
        self.scene_settings = []
        self.scene_objects = []

        # check contact
        self.check_contact = False
        self.contact_error = False

    def set_controller_camera_pose(self, x, y, z, yaw, pitch):
        self.renderer_controller.set_camera_position(x, y, z)
        self.renderer_controller.set_camera_rotation(yaw, pitch)
        self.renderer_controller.render()

    def get_global_mesh(self, obj):
        final_vs = []; final_fs = []; vid = 0;
        # if obj:
        for l in obj.get_links():
            vs = []
            for s in l.get_collision_shapes():
                v = np.array(s.convex_mesh_geometry.vertices, dtype=np.float32)
                f = np.array(s.convex_mesh_geometry.indices, dtype=np.uint32).reshape(-1, 3)
                vscale = s.convex_mesh_geometry.scale
                v[:, 0] *= vscale[0]; v[:, 1] *= vscale[1]; v[:, 2] *= vscale[2];
                ones = np.ones((v.shape[0], 1), dtype=np.float32)
                v_ones = np.concatenate([v, ones], axis=1)
                transmat = s.pose.to_transformation_matrix()
                v = (v_ones @ transmat.T)[:, :3]
                vs.append(v)
                final_fs.append(f + vid)
                vid += v.shape[0]
            if len(vs) > 0:
                vs = np.concatenate(vs, axis=0)
                ones = np.ones((vs.shape[0], 1), dtype=np.float32)
                vs_ones = np.concatenate([vs, ones], axis=1)
                transmat = l.get_pose().to_transformation_matrix()
                vs = (vs_ones @ transmat.T)[:, :3]
                final_vs.append(vs)
        final_vs = np.concatenate(final_vs, axis=0)
        final_fs = np.concatenate(final_fs, axis=0)
        return final_vs, final_fs

    def sample_pc(self, v, f, n_points = 10000):
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)
        return points

    def load_object(self, urdf, material, state='closed'):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.object = loader.load(urdf, {"material": material})
        #self.object = loader.load(urdf, material)
        pose = Pose([self.object_position_offset, 0, 0], [1, 0, 0, 0])
        self.object.set_root_pose(pose)

        # compute link actor information
        # cur_occlusion_object_lids = [l.get_id() for l in (o.get_links() for o in env.scene_objects)]
        # self.all_link_ids = [l.get_id() for l in self.object.get_links()]

        # self.all_link_ids = [l.get_id() for l in (o.get_links() for o in self.scene_objects)]
        self.movable_link_ids = []
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                self.movable_link_ids.append(j.get_child_link().get_id())
        if self.flog is not None:
            # self.flog.write('All Actor Link IDs: %s\n' % str(self.all_link_ids))
            self.flog.write('All Movable Actor Link IDs: %s\n' % str(self.movable_link_ids))

        # set joint property
        for joint in self.object.get_joints():
            joint.set_drive_property(stiffness=0, damping=10)

        # set initial qpos
        joint_angles = []
        self.joint_angles_lower = []
        self.joint_angles_upper = []
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                l = process_angle_limit(j.get_limits()[0, 0])
                self.joint_angles_lower.append(float(l))
                r = process_angle_limit(j.get_limits()[0, 1])
                self.joint_angles_upper.append(float(r))
                if state == 'closed':
                    joint_angles.append(float(l))
                elif state == 'open':
                    joint_angles.append(float(r))
                elif state == 'random-middle':
                    joint_angles.append(float(get_random_number(l, r)))
                elif state == 'middle':
                    # joint_angles.append(float((l + r) / 2))
                    joint_angles.append(float(l*19/20 + r/20))
                elif state == 'random-closed-middle':
                    if np.random.random() < 0.5:
                        joint_angles.append(float(get_random_number(l, r)))
                    else:
                        if np.random.random() < 0.9:
                            joint_angles.append(float(l*9/10 + r/10))
                        else:
                            joint_angles.append(float(l))
                else:
                    raise ValueError('ERROR: object init state %s unknown!' % state)
        self.object.set_qpos(joint_angles)
        self.scene_objects.append(self.object)
        final_vs, _ = self.get_global_mesh(self.object)
        z_min = np.min(final_vs[:, 2])
        self.ground = self.scene.add_ground(z_min, render=False)
        print(z_min)
        self.ground_z = z_min
        # self.ground_z_min = z_min-0.01
        return joint_angles

    def get_urdf_bounding_box(self, urdf_file):
        # create a Sapien scene and pose loader
        scene = self.engine.create_scene()
        pose_loader = scene.create_urdf_loader()

        # load the URDF file
        actor = pose_loader.load(urdf_file)
        final_vs = []; final_fs = []; vid = 0;
        for l in actor.get_links():
            vs = []
            for s in l.get_collision_shapes():
                v = np.array(s.convex_mesh_geometry.vertices, dtype=np.float32)
                f = np.array(s.convex_mesh_geometry.indices, dtype=np.uint32).reshape(-1, 3)
                vscale = s.convex_mesh_geometry.scale
                v[:, 0] *= vscale[0]; v[:, 1] *= vscale[1]; v[:, 2] *= vscale[2];
                ones = np.ones((v.shape[0], 1), dtype=np.float32)
                v_ones = np.concatenate([v, ones], axis=1)
                transmat = s.pose.to_transformation_matrix()
                v = (v_ones @ transmat.T)[:, :3]
                vs.append(v)
                final_fs.append(f + vid)
                vid += v.shape[0]
            if len(vs) > 0:
                vs = np.concatenate(vs, axis=0)
                ones = np.ones((vs.shape[0], 1), dtype=np.float32)
                vs_ones = np.concatenate([vs, ones], axis=1)
                transmat = l.get_pose().to_transformation_matrix()
                vs = (vs_ones @ transmat.T)[:, :3]
                final_vs.append(vs)
        final_vs = np.concatenate(final_vs, axis=0)
        final_fs = np.concatenate(final_fs, axis=0)

        vertices = final_vs

        # calculate the minimum and maximum x- and y-coordinates
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()

        print((x_min, y_min, z_min, x_max, y_max, z_max))

        return (x_min, y_min, z_min, x_max, y_max, z_max)

    def load_occlusion(self, urdf, item, material, state='closed'):
        loader = self.scene.create_urdf_loader()
        loader.scale = item['init_scale']
        loader.fix_root_link = True
        new_obj = loader.load(urdf, {"material": material})
        if 'init_x' not in item: item['init_x'] = 0
        if 'init_y' not in item: item['init_y'] = 0
        if 'init_z' not in item: item['init_z'] = 0
        if 'init_quat' not in item: item['init_quat'] = [1, 0, 0, 0]

        if 'joint_angles' not in item:
            joint_angles = []
            # if new_obj:
            for j in new_obj.get_joints():
                if j.get_dof() == 1:
                    l = process_angle_limit(j.get_limits()[0, 0])
                    r = process_angle_limit(j.get_limits()[0, 1])
                    if state == 'closed':
                        joint_angles.append(float(l))
                    elif state == 'locked':
                        joint_angles.append(float(l))
                    elif state == 'open':
                        joint_angles.append(float(r))
                    elif state == 'middle':
                        joint_angles.append(float((l + r) / 2))
                    elif state == 'random-middle':
                        joint_angles.append(float(get_random_number(l, r)))
                    elif state == 'random-closed-middle':
                        if np.random.random() < 0.5:
                            joint_angles.append(float(get_random_number(l, r)))
                        else:
                            joint_angles.append(float(l))
                    else:
                        raise ValueError('ERROR: object init state %s unknown!' % state)
            item['joint_angles'] = joint_angles
        new_obj.set_qpos(item['joint_angles']) # set_object_joint_angles
        pose = Pose([item['init_x'], item['init_y'], item['init_z']], item['init_quat'])
        new_obj.set_root_pose(pose)
        print('[env::load_object] loaded: %s' % str(item))

        self.scene_settings.append(item)
        self.scene_objects.append(new_obj)
        
        return item['joint_angles']


    def set_object_joint_angles(self, joint_angles):
        self.object.set_qpos(joint_angles)

    def set_target_object_part_actor_id(self, actor_id):
        # self.all_link_ids = [l.get_id() for l in (o.get_links() for o in self.scene_objects)]
        self.all_link_ids = []
        for j in self.scene_objects:
            for l in j.get_links():
                self.all_link_ids.append(l.get_id())
        # self.movable_link_ids = []
        # for j in self.object.get_joints():
        #     if j.get_dof() == 1:
        #         self.movable_link_ids.append(j.get_child_link().get_id())
        if self.flog is not None:
            self.flog.write('All Actor Link IDs: %s\n' % str(self.all_link_ids))
            # self.flog.write('All Movable Actor Link IDs: %s\n' % str(self.movable_link_ids))
        if self.flog is not None:
            self.flog.write('Set Target Object Part Actor ID: %d\n' % actor_id)
        self.target_object_part_actor_id = actor_id
        self.non_target_object_part_actor_id = list(set(self.all_link_ids) - set([actor_id]))

        # get the link handler
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == actor_id:
                    self.target_object_part_actor_link = j.get_child_link()
        
        # moniter the target joint
        idx = 0
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == actor_id:
                    self.target_object_part_joint_id = idx
                idx += 1

    def get_object_qpos(self):
        return self.object.get_qpos()

    def get_scene_qpos(self):
        # scene_qpos = self.object.get_qpos()
        scene_qpos = np.array([])
        for obj in self.scene_objects:
            scene_qpos=np.append(scene_qpos, np.array(obj.get_qpos()))
        return scene_qpos

    def get_target_part_qpos(self):
        qpos = self.object.get_qpos()
        return float(qpos[self.target_object_part_joint_id])
    
    def get_target_part_pose(self):
        return self.target_object_part_actor_link.get_pose()

    def start_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, robot_arm_actor_ids, strict):
        self.check_contact = True
        self.check_contact_strict = strict
        self.first_timestep_check_contact = True
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids
        self.robot_arm_actor_ids = robot_arm_actor_ids
        self.contact_error = False

    def end_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, robot_arm_actor_ids, strict):
        self.check_contact = False
        self.check_contact_strict = strict
        self.first_timestep_check_contact = False
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids
        self.robot_arm_actor_ids = robot_arm_actor_ids

    def get_material(self, static_friction, dynamic_friction, restitution):
        return self.engine.create_physical_material(static_friction, dynamic_friction, restitution)

    def render(self):
        if self.show_gui and (not self.window):
            self.window = True
            self.renderer_controller.show_window()
        self.scene.update_render()
        if self.show_gui and (self.current_step % self.render_rate == 0):
            self.renderer_controller.render()

    def step(self):
        self.current_step += 1
        self.scene.step()
        if self.check_contact:
            if not self.check_contact_is_valid():
                raise ContactError()
                # print('contact error here!')

    # check the first contact: only gripper links can touch the target object part link
    def check_contact_is_valid(self):
        self.contacts = self.scene.get_contacts()
        contact = False; valid = False; 
        for c in self.contacts:
            aid1 = c.actor1.get_id()
            aid2 = c.actor2.get_id()
            has_impulse = False
            for p in c.points:
                if abs(p.impulse @ p.impulse) > 1e-4:
                    has_impulse = True
                    break
            if has_impulse:
                if (aid1 in self.robot_gripper_actor_ids and aid2 == self.target_object_part_actor_id) or \
                   (aid2 in self.robot_gripper_actor_ids and aid1 == self.target_object_part_actor_id):
                    contact, valid = True, True
                # elif (aid1 in self.robot_gripper_actor_ids and aid2 in self.non_target_object_part_actor_id) or \
                #    (aid2 in self.robot_gripper_actor_ids and aid1 in self.non_target_object_part_actor_id):
                elif (aid1 in self.robot_gripper_actor_ids or aid2 in self.robot_gripper_actor_ids):
                    if (aid1 in self.robot_gripper_actor_ids and aid2 in self.robot_gripper_actor_ids):
                        pass
                    elif (aid1 == self.robot_hand_actor_id or aid2 == self.robot_hand_actor_id):
                        pass
                    elif (aid1 in self.robot_arm_actor_ids or aid2 in self.robot_arm_actor_ids):
                        pass
                    else:
                    # if self.check_contact_strict:
                        print('gripper & non-target contact error!')
                        return False
                    # else:
                        # contact, valid = True, True
                elif (aid1 == self.robot_hand_actor_id or aid2 == self.robot_hand_actor_id):
                    # if not self.check_contact_strict:
                        # print('robothand_error_contact')
                        # return False
                    # else:
                    # if (aid1 == self.target_object_part_actor_id or aid2 == self.target_object_part_actor_id):
                    #     contact, valid = True, True
                    # elif (aid1 not in self.robot_gripper_actor_ids and aid2 not in self.robot_gripper_actor_ids):
                    # elif (aid1 in self.non_target_object_part_actor_id or aid2 in self.non_target_object_part_actor_id):
                    if (aid1 in self.robot_arm_actor_ids or aid2 in self.robot_arm_actor_ids):
                        pass
                    else:
                        if (aid2 == self.target_object_part_actor_id or aid2 == self.target_object_part_actor_id):
                            contact, valid = True, True
                        else:
                            print('hand contact error!')
                            return False
                elif (aid1 in self.robot_arm_actor_ids or aid2 in self.robot_arm_actor_ids):
                    # if not self.check_contact_strict:
                        # print('robothand_error_contact')
                        # return False
                    # else:
                    # if (aid1 == self.target_object_part_actor_id or aid2 == self.target_object_part_actor_id):
                    #     # contact, valid = True, True
                    #     print('robotarm_error_contact')
                    #     return False
                    # # elif (aid1 not in self.robot_gripper_actor_ids and aid2 not in self.robot_gripper_actor_ids):
                    # elif (aid1 in self.non_target_object_part_actor_id or aid2 in self.non_target_object_part_actor_id):
                    # if (aid1 in self.robot_gripper_actor_ids or aid2 in self.robot_gripper_actor_ids):
                    #     pass
                    # elif (aid1 == self.robot_hand_actor_id or aid2 == self.robot_hand_actor_id):
                    if (aid1 in self.robot_arm_actor_ids and aid2 in self.robot_arm_actor_ids):
                        pass
                    else:
                        print('robotarm_error_contact')
                        return False
                '''
                # starting pose should have no collision at all
                if (aid1 in self.robot_gripper_actor_ids or aid1 == self.robot_hand_actor_id or aid1 in self.robot_arm_actor_ids or\
                    aid2 in self.robot_gripper_actor_ids or aid2 == self.robot_hand_actor_id or aid1 in self.robot_arm_actor_ids) and self.first_timestep_check_contact:
                    # if (aid1 == self.target_object_part_actor_id or aid2 == self.target_object_part_actor_id):
                    #     print('first_timestep_check_contact error')
                    #     return False
                    # elif (aid1 in self.non_target_object_part_actor_id or aid2 in self.non_target_object_part_actor_id):
                    #     print('first_timestep_check_contact error')
                    #     return False
                    if ((aid1 in self.robot_gripper_actor_ids or aid1 == self.robot_hand_actor_id or aid1 in self.robot_arm_actor_ids) and\
                    (aid2 in self.robot_gripper_actor_ids or aid2 == self.robot_hand_actor_id or aid2 in self.robot_arm_actor_ids)):
                        pass
                    else:
                        print('first_timestep contact error!')
                        return False
                '''
        self.first_timestep_check_contact = False
        # if contact and valid:
        #     self.check_contact = False
        return True

    def close_render(self):
        if self.window:
            self.renderer_controller.hide_window()
        self.window = False
    
    def wait_to_start(self):
        print('press q to start\n')
        while not self.renderer_controller.should_quit:
            self.scene.update_render()
            if self.show_gui:
                self.renderer_controller.render()

    def close(self):
        if self.show_gui:
            self.renderer_controller.set_current_scene(None)
        self.scene = None


