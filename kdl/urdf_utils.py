import numpy as np
import tensorflow as tf
from .utils.import_pykdl import *
from .utils.urdf_parser_py.urdf import URDF
from .joint import JointType, Joint, Link
from .frame import Frame, Rotation
from .segment import Segment
from .chain import Chain
from .rbi import RigidBodyInertia, RotationalInertia
import trimesh

def joint_list_to_kdl(q):
    if q is None:
        return None
    if type(q) == np.matrix and q.shape[1] == 0:
        q = q.T.tolist()[0]
    q_kdl = kdl.JntArray(len(q))
    for i, q_i in enumerate(q):
        q_kdl[i] = q_i
    return q_kdl

def kdl_to_mat(m):
    mat =  np.mat(np.zeros((m.rows(), m.columns())))
    for i in range(m.rows()):
        for j in range(m.columns()):
            mat[i,j] = m[i,j]
    return mat

def kdl_dyn_from_kdl_chain(chain):
	return kdl.ChainDynParam(chain, kdl.Vector.Zero())

def kdl_inertia_matrix_from_urdf(urdf, q):
	pykdl_tree = kdl_tree_from_urdf_model(urdf)
	pykdl_chain = pykdl_tree.getChain(urdf.links[0].name, urdf.links[-1].name)
	dyn_pykdl = kdl.ChainDynParam(pykdl_chain, kdl.Vector.Zero())
	H_pykdl = kdl.JntSpaceInertiaMatrix(pykdl_chain.getNrOfJoints())
	dyn_pykdl.JntToMass(joint_list_to_kdl(q[:pykdl_chain.getNrOfJoints()]), H_pykdl)
	return kdl_to_mat(H_pykdl), dyn_pykdl, H_pykdl

def kdl_chain_from_urdf(urdf):
	pykdl_tree = kdl_tree_from_urdf_model(urdf)
	return pykdl_tree.getChain(urdf.links[0].name, urdf.links[-1].name)


## Change all Joint.None to getattr(Joint, "None")
def euler_to_quat(r, p, y):
	sr, sp, sy = np.sin(r / 2.0), np.sin(p / 2.0), np.sin(y / 2.0)
	cr, cp, cy = np.cos(r / 2.0), np.cos(p / 2.0), np.cos(y / 2.0)
	return [sr * cp * cy - cr * sp * sy,
			cr * sp * cy + sr * cp * sy,
			cr * cp * sy - sr * sp * cy,
			cr * cp * cy + sr * sp * sy]

def quat_to_rot(q):
	x, y, z, w = q

	r = np.array([
		[1.- 2. * (y ** 2 + z ** 2), 2 * (x * y - z * w),            2 * (x * z + y * w)],
		[2 * (x * y + z * w),        1.- 2. * (x ** 2 + z ** 2),     2 * (y * z - x * w)],
		[2 * (x * z - y * w),        2 * (y * z + x * w),            1.- 2. * (y ** 2 + x ** 2)]
	])
	return r

def urdf_pose_to_kdl_frame(pose):
	pos = [0., 0., 0.]
	rot = [0., 0., 0.]
	if pose is not None:
		if pose.position is not None:
			pos = pose.position
		if pose.rotation is not None:
			rot = pose.rotation
	return kdl.Frame(kdl.Rotation.Quaternion(*euler_to_quat(*rot)),
					 kdl.Vector(*pos))


def urdf_pose_to_tk_frame(pose):
	pos = [0., 0., 0.]
	rot = [0., 0., 0.]

	if pose is not None:
		if pose.position is not None:
			pos = pose.position
		if pose.rotation is not None:
			rot = pose.rotation

	return Frame(p=tf.constant(pos, dtype=tf.float32),
				 m=tf.constant(quat_to_rot(euler_to_quat(*rot)), dtype=tf.float32))


def urdf_joint_to_kdl_joint(jnt):
	origin_frame = urdf_pose_to_kdl_frame(jnt.origin)
	if jnt.joint_type == 'fixed':
		return kdl.Joint(jnt.name, getattr(kdl.Joint, "None"))
	axis = kdl.Vector(*[float(s) for s in jnt.axis])
	if jnt.joint_type == 'revolute':
		return kdl.Joint(jnt.name, origin_frame.p,
						 origin_frame.M * axis, kdl.Joint.RotAxis)
	if jnt.joint_type == 'continuous':
		return kdl.Joint(jnt.name, origin_frame.p,
						 origin_frame.M * axis, kdl.Joint.RotAxis)
	if jnt.joint_type == 'prismatic':
		return kdl.Joint(jnt.name, origin_frame.p,
						 origin_frame.M * axis, kdl.Joint.TransAxis)
	print("Unknown joint type: %s." % jnt.joint_type)
	return kdl.Joint(jnt.name, getattr(kdl.Joint, "None"))


def urdf_joint_to_tk_joint(jnt):
	origin_frame = urdf_pose_to_tk_frame(jnt.origin)

	if jnt.joint_type == 'revolute':
		axis = tf.constant(jnt.axis, dtype=tf.float32)
		return Joint(JointType.RotAxis, origin=origin_frame.p ,
				 axis=tf.matmul(origin_frame.m, tf.expand_dims(axis, 1))[:,0], name=jnt.name,
					 limits=jnt.limit), origin_frame

	if jnt.joint_type == 'fixed' or jnt.joint_type == 'prismatic':
		return Joint(JointType.NoneT, name=jnt.name), origin_frame

	print("Unknown joint type: %s." % jnt.joint_type)


def urdf_link_to_tk_link(lnk):
	if lnk.inertial is not None and lnk.inertial.origin is not None:
		return Link(frame=urdf_pose_to_tk_frame(lnk.inertial.origin), mass=lnk.inertial.mass)
	else:
		return Link(frame=urdf_pose_to_tk_frame(None), mass=1.)


def urdf_inertial_to_kdl_rbi(i):
	origin = urdf_pose_to_kdl_frame(i.origin)
	rbi = kdl.RigidBodyInertia(i.mass, origin.p,
							   kdl.RotationalInertia(i.inertia.ixx,
													 i.inertia.iyy,
													 i.inertia.izz,
													 i.inertia.ixy,
													 i.inertia.ixz,
													 i.inertia.iyz))
	return origin.M * rbi

def urdf_inertial_to_tk_rbi(i):
	origin = urdf_pose_to_tk_frame(i.origin)
	rbi = RigidBodyInertia(m=i.mass, c=origin.p, # c = ? or h ?
							   I_c=RotationalInertia(Ixx=i.inertia.ixx, # I_c = ? or I_r ?
													 Iyy=i.inertia.iyy,
													 Izz=i.inertia.izz,
													 Ixy=i.inertia.ixy,
													 Ixz=i.inertia.ixz,
													 Iyz=i.inertia.iyz))
	return  rbi * Rotation(origin.m)


# Returns a PyKDL.Tree generated from a urdf_parser_py.urdf.URDF object.
def kdl_tree_from_urdf_model(urdf):
	root = urdf.get_root()
	tree = kdl.Tree(root)

	def add_children_to_tree(parent):
		if parent in urdf.child_map:
			for joint, child_name in urdf.child_map[parent]:
				for lidx, link in enumerate(urdf.links):
					if child_name == link.name:
						child = urdf.links[lidx]
						if child.inertial is not None:
							kdl_inert = urdf_inertial_to_kdl_rbi(child.inertial)
						else:
							kdl_inert = kdl.RigidBodyInertia()
						for jidx, jnt in enumerate(urdf.joints):
							if jnt.name == joint:
								kdl_jnt = urdf_joint_to_kdl_joint(urdf.joints[jidx])
								kdl_origin = urdf_pose_to_kdl_frame(
									urdf.joints[jidx].origin)
								kdl_sgm = kdl.Segment(child_name, kdl_jnt,
													  kdl_origin, kdl_inert)
								tree.addSegment(kdl_sgm, parent)
								add_children_to_tree(child_name)

	add_children_to_tree(root)
	return tree


def tk_tree_from_urdf_model(urdf):
	raise NotImplementedError
	root = urdf.get_root()
	tree = kdl.Tree(root)

	def add_children_to_tree(parent):
		if parent in urdf.child_map:
			for joint, child_name in urdf.child_map[parent]:
				for lidx, link in enumerate(urdf.links):
					if child_name == link.name:
						for jidx, jnt in enumerate(urdf.joints):
							if jnt.name == joint:
								tk_jnt, tk_origin = urdf_joint_to_tk_joint(
									urdf.joints[jidx])
								tk_origin = urdf_pose_to_tk_frame(urdf.joints[jidx].origin)

								tree.segments += [Segment(joint=tk_jnt, f_tip=tk_origin,
													 child_name=child_name)]

								tree.addSegment(kdl_sgm, parent)
								add_children_to_tree(child_name)

	add_children_to_tree(root)
	return tree


def tk_chain_from_urdf(urdf, root=None, tip=None,
							  load_collision=False, mesh_path=None):

	if mesh_path is not None and mesh_path[-1] != '/': mesh_path += '/'

	root = urdf.get_root() if root is None else root
	segments = []

	chain = None if tip is None else urdf.get_chain(root, tip)[1:]

	def add_children_to_chain(parent, segments, chain=None):
		if parent in urdf.child_map:
			# print "parent:", parent
			# print "childs:", urdf.child_map[parent]
			#
			if chain is not None:
				childs = [child for child in urdf.child_map[parent] if child[1] in chain]
				if len(childs):
					joint, child_name = childs[0]
				else:
					return
			else:
				if not len(urdf.child_map[parent]) < 2:
					print("Robot is not a chain, taking first branch")

				joint, child_name = urdf.child_map[parent][0]
			for joint, child_name in urdf.child_map[parent]:
				for lidx, link in enumerate(urdf.links):
					if child_name == link.name:
						child = urdf.links[lidx]
						tk_rbi = urdf_inertial_to_tk_rbi(child.inertial)

						for jidx, jnt in enumerate(urdf.joints):
							if jnt.name == joint and jnt.joint_type in ['revolute', 'fixed', 'prismatic']:
								tk_jnt, tk_origin = urdf_joint_to_tk_joint(urdf.joints[jidx])
								tk_origin = urdf_pose_to_tk_frame(urdf.joints[jidx].origin)
								tk_lnk = urdf_link_to_tk_link(urdf.link_map[child_name])

								if load_collision and urdf.link_map[child_name].collision is not None:

									filename = mesh_path + \
											   urdf.link_map[child_name].collision.geometry.filename.split('/')[-1]

									# filename = filename[:-14] + '.STL'
									# tk_lnk.collision_mesh = mesh.Mesh.from_file(filename)
									tk_lnk.collision_mesh = trimesh.load(filename)

								segments += [Segment(
									joint=tk_jnt, f_tip=tk_origin, I=tk_rbi, child_name=child_name, link=tk_lnk
								)]


								add_children_to_chain(child_name, segments, chain)

	add_children_to_chain(root, segments, chain)
	return Chain(segments)


def kdl_tree_from_file(file):
	return kdl_tree_from_urdf_model(URDF.from_xml_file(file))


def urdf_from_file(file):
	return URDF.from_xml_file(file)