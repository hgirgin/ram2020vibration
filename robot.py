import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import networkx as nx

import matplotlib
import matplotlib.patches as patches

from costs import *

class Robot():
    def __init__(self,l1, l2):
        self.l1 = l1
        self.l2 = l2
        self.theta1 = 0.
        self.theta2 = 0.
        self.obstacles = []
        self.R = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)],[np.sin(np.pi/2), np.cos(np.pi/2)]])
        
    def set_dof(self,theta):
        self.theta1 = theta[0]
        self.theta2 = theta[1]
        
    def rotate90(self,vec):
        return np.dot(self.R, vec)
    
    def add_obstacle(self, x, y, r):
        obstacle = dict()
        obstacle['rad'] = r
        obstacle['pos'] = np.array([x,y])
        self.obstacles += [obstacle]
        
    def check_collision_single(self, obstacle):    
        c = obstacle['pos'][:,None]
        r = obstacle['rad']
        b1 = c - self.p0
        b2 = c - self.p1
        self.res1 = np.linalg.solve(self.A1,b1)
        self.res2 = np.linalg.solve(self.A2,b2)
        if self.res1[0] >= 0. and self.res1[0] <= 1.:
            if np.abs(self.res1[1]) <= r:
                return True
        else:
            if np.linalg.norm(self.p0-c) < r or np.linalg.norm(self.p1-c) < r:
                return True
        
        if self.res2[0] >= 0. and self.res2[0] <= 1.:
            if np.abs(self.res2[1]) <= r:
                return True
        else:
            if np.linalg.norm(self.p1-c) < r or np.linalg.norm(self.p2-c) < r:
                return True
            
        return False
        
        
    def normalize(self,vec):
        return vec/np.linalg.norm(vec)
        
    def check_collision(self):
        self.p0 = np.array([0.,0.])[:,None]
        self.p1 = np.array([self.l1*np.cos(self.theta1),self.l1*np.sin(self.theta1)])[:,None]
        self.p2 = np.array([self.p1[0,0] + self.l2*np.cos(self.theta1+self.theta2),self.p1[1,0] + self.l2*np.sin(self.theta1+self.theta2)])[:,None]
        self.v1 = self.normalize(self.rotate90(self.p1-self.p0))
        self.v2 = self.normalize(self.rotate90(self.p2-self.p1))
        self.A1 = np.hstack([self.p1-self.p0, -self.v1])
        self.A2 = np.hstack([self.p2-self.p1, -self.v2])
        
        for obstacle in self.obstacles:
            if self.check_collision_single(obstacle):
                return True
            
        return False
        
        
        
    def plot(self):
        x = [0., self.l1*np.cos(self.theta1), self.l1*np.cos(self.theta1) + self.l2*np.cos(self.theta1+ self.theta2)]
        y = [0., self.l1*np.sin(self.theta1), self.l1*np.sin(self.theta1) + self.l2*np.sin(self.theta1+ self.theta2)]
        plt.plot(x,y,'b',linewidth=5)
        plt.plot(x[:-1],y[:-1],'oy',markersize=3 )
        fig = plt.gcf()
        ax = fig.gca()
        for obstacle in self.obstacles:
            r = obstacle['rad']
            p = obstacle['pos']
            circle = plt.Circle(p, r, color='r')
            ax.add_artist(circle)
            
        #add base
        rect = patches.Rectangle((-0.35,-0.35),0.7,0.7,linewidth=1,edgecolor='g',facecolor='g')
        ax.add_patch(rect)
        
        ax.axis('equal')
        ax.set(xlim=(-2*self.l1-2*self.l2,2*self.l1+2*self.l2), ylim=( -2*self.l1-2*self.l2,2*self.l1+2*self.l2))
        
        
    def sample_state(self, N=1, get_valid = False):
        samples = []
        samples_status = []
        while len(samples) < N:
            sample = np.random.rand(2)*np.pi*2
            self.set_dof(sample)
            sample_status = self.check_collision()
            if get_valid:
                if sample_status is True:
                    continue
            samples.append(sample)
            samples_status.append(sample_status)
            
        return np.array(samples), np.array(samples_status)
    
class RRT():
    def __init__(self, D, sampler, col_checker, interpolator):
        self.D = D
        self.sampler = sampler
        self.col_checker = col_checker
        self.interpolator = interpolator
        self.samples = [] 
        
    def check_collision(self, sample):
        return self.col_checker.check_collision(sample)
    
    def sample(self, get_valid = True):
        status = True
        if get_valid:
            while status is True:
                sample =  self.sampler.sample()
                status = self.check_collision(sample.flatten())
        else:
                sample =  self.sampler.sample()            
        return sample
    
    def interpolate(self, sample1, sample2, N):
        return self.interpolator.interpolate(sample1, sample2, N)
               
    def extend(self, cur_index, sample1, sample2, step_length = 0.1):
        #state1, state2 = self.correct_manifold(sample1, sample2)
        state1, state2 = sample1.copy(), sample2.copy()
        dist = np.linalg.norm(state2-state1)
        N = int(dist/step_length) + 2
        state_list = self.interpolate(state1, state2, N)
        next_states = []
        for state in state_list:
            if self.check_collision(state):
                #print('collision')
                #print(state)
                break
            next_states += [(state)]
            
        #add the nodes to the graph
        for next_state in next_states[1:]:
            next_index = self.G.number_of_nodes()
            self.G.add_nodes_from([next_index])
            self.G.add_edge(cur_index, next_index)
            self.samples += [next_state]
            cur_index = next_index
            
        return next_states

    def find_nearest(self, sample1, sample_set, n = 1):
        #find a nearest node
        if len(sample_set) > 1:
            index = np.argpartition(np.linalg.norm(sample1-sample_set,axis=1),n)[0:n]
            return index,sample_set[index]
        else:
            return [0], sample_set[0]  
    
    def init_plan(self, init_state, goal_state):
        self.G = nx.Graph()
        self.nodes = [0]
        self.edges = []
        self.samples = [init_state]
        self.G.add_node(0)
        self.init_state = init_state.copy()
        self.goal_state = goal_state.copy()
    
    def step_plan(self):
        success = False
        #sample random state
        self.random_sample = self.sample()
        #set_q_std(random_sample)
        #find a nearest node
        nearest_index, nearest_sample = self.find_nearest(self.random_sample, np.array(self.samples))

        #extend to the random state
        self.next_states = self.extend(nearest_index[0], nearest_sample.flatten(), self.random_sample.flatten() )
        
        #check distance with goal state
        nearest_index, nearest_sample = self.find_nearest(self.goal_state, np.array(self.samples))
        
        #extend to the goal state
        clear_output()
        #print('Reached a random state...')
        #print(nearest_sample)
        #input()
        self.next_states_goal = self.extend(nearest_index[0], nearest_sample.flatten(), self.goal_state.flatten())
        #print(next_states)
        if len(self.next_states_goal) == 0: return False
        clear_output()
        #print('Reached a goal state...')
        #print(self.next_states_goal[-2:])
        #print(self.goal_state)
        #input()
        
        if np.linalg.norm(self.next_states_goal[-1] - self.goal_state)< 0.001:
            print('Solution is found!')
            success = True
        return success,0
        
    def plan(self, init_state, goal_state):
        self.init_plan(init_state, goal_state)
        success = False
        self.num_plan = 0
        self.nfevs = 0
        while success is False:
            success, nfev = self.step_plan()
            self.nfevs += nfev
            self.num_plan += 1
            print('Planning...')
        #clear_output()
        #find the path
        path = nx.dijkstra_path(self.G, 0, self.G.number_of_nodes()-1)
        return path, self.nfevs
        
    def plot_graph(self):
        edges = self.G.edges().keys()
        segs = []
        for edge in edges:
            node1 = self.samples[edge[0]]
            node2 = self.samples[edge[1]]
            segs.append([[node1[0], node1[1]],[node2[0],node2[1]]])
        ln_coll = matplotlib.collections.LineCollection(segs)
        ax = plt.gca()
        ax.add_collection(ln_coll)
        plt.axis([0,2*np.pi,0,2*np.pi])
        
    def shortcut_path(self, path_in, step_length=0.1):
        #simple shortcutting algo trying to iteratively remove nodes from the path
        path = path_in.copy()
        while len(path) > 2:
            idx_remove = -1
            for idx in range(len(path)-2):
                #try to remove node(idx+1) from path: Is interpolation between node(idx) and node(idx+2) free?
                node1 = self.samples[path[idx]]
                node2 = self.samples[path[idx+2]]
                dist = np.linalg.norm(node2-node1)
                N = int(dist/step_length) + 2
                state_list = self.interpolate(node1, node2, N)
                free = True
                for state in state_list:
                    if self.check_collision(state):
                        free = False
                        break
                        
                if free is True:
                    idx_remove = idx+1
                    break
                
            if idx_remove == -1:
                break
            del path[idx_remove]
        return path

class BiRRT(RRT):
    def init_plan(self, init_state, goal_state):
        self.G0, self.GT = nx.Graph(), nx.Graph()
        self.G = self.G0
        self.nodes = [0]
        self.edges = []
        self.samples0 = [init_state]
        self.samplesT = [goal_state]
        self.samples = self.samples0
        self.G0.add_node(0)
        self.GT.add_node(0)
        self.init_state = init_state.copy()
        self.goal_state = goal_state.copy()

    def step_plan(self):
        success = False

        # Sample random configs
        q_rand = self.sample()[0]

        # find a nearest node
        self.G = self.G0
        self.samples = self.samples0
        nearest_index_, nearest_sample = self.find_nearest(q_rand, np.array(self.samples))
        nearest_index = nearest_index_[0]

        # Extend to this config
        self.next_states= self.extend(nearest_index, nearest_sample.flatten(), q_rand)
        self.q_reach_a = self.next_states[-1]

        # Switch the tree
        self.G = self.GT
        self.samples = self.samplesT
        nearest_index_, nearest_sample = self.find_nearest(self.q_reach_a, np.array(self.samples))
        nearest_index = nearest_index_[0]

        # Extend to this config
        self.next_states= self.extend(nearest_index, nearest_sample.flatten(), self.q_reach_a)
        self.q_reach_b = self.next_states[-1]

        # check
        if np.linalg.norm(self.q_reach_a - self.q_reach_b) < 1e-3:
            print('Goal is reached!')
            success = True
            return success

        # repeat after swapping the tree
        # Sample random configs
        q_rand = self.sample()[0]

        # find a nearest node
        self.G = self.GT
        self.samples = self.samplesT
        nearest_index_, nearest_sample = self.find_nearest(q_rand, np.array(self.samples))
        nearest_index = nearest_index_[0]

        # Extend to this config
        self.next_states= self.extend(nearest_index, nearest_sample.flatten(), q_rand)
        self.q_reach_a = self.next_states[-1]

        # Switch the tree
        self.G = self.G0
        self.samples = self.samples0
        nearest_index_, nearest_sample = self.find_nearest(self.q_reach_a, np.array(self.samples))
        nearest_index = nearest_index_[0]

        # Extend to this config
        self.next_states= self.extend(nearest_index, nearest_sample.flatten(), self.q_reach_a)
        self.q_reach_b = self.next_states[-1]

        # check
        if np.linalg.norm(self.q_reach_a - self.q_reach_b) < 1e-3:
            print('Goal is reached!')
            success = True
            return success
        return success

    def plan(self, q_init, q_goal, max_extension_steps=500, n_retry=3):
        success = False
        total_extension = 0
        total_projection = 0
        for i in range(n_retry):
            self.init_plan(q_init, q_goal)

            ext_steps = 0
            while ext_steps < max_extension_steps:
                success = self.step_plan()
                ext_steps += 1
                if success:
                    break

            total_extension += ext_steps
            if success: break
        self.retry = i
        # clear_output()

        if success is False:
            print("No solution found!")
            return [], 0, 0, False, self.retry
        print("Solution found!")

        # find the path
        path1 = nx.dijkstra_path(self.G0, 0, self.G0.number_of_nodes() - 1)
        path2 = nx.dijkstra_path(self.GT, 0, self.GT.number_of_nodes() - 1)
        self.traj = []
        for i in path1:
            self.traj += [self.samples0[i]]
        for i in path2[::-1]:
            self.traj += [self.samplesT[i]]

        return self.traj, ext_steps, success
    
    def shortcut_path_old(self, traj, step_length=0.1):
        # simple shortcutting algo trying to iteratively remove nodes from the path
        path = list(np.arange(len(traj)))
        while len(path) > 2:
            idx_remove = -1
            for idx in range(len(path) - 2):
                # try to remove node(idx+1) from path: Is interpolation between node(idx) and node(idx+2) free?
                node1 = traj[path[idx]]
                node2 = traj[path[idx + 2]]
                dist = np.linalg.norm(node2 - node1)
                N = int(dist / step_length) + 2
                state_list = self.interpolate(node1, node2, N)
                free = True
                for state in state_list:
                    if self.check_collision(state) is True:
                        free = False
                        break

                if free is True:
                    idx_remove = idx + 1
                    break

            if idx_remove == -1:
                break
            del path[idx_remove]
        return path
    def shortcut_path(self, traj, step_length=0.05, max_try = 100):
        # simple shortcutting algo trying to iteratively remove nodes from the path
        path = list(np.arange(len(traj)))
        i = 0
        while i < max_try and len(path) > 2:
            i+=1
            idx1 = np.random.randint(len(path)-2)
            idx2 = np.random.randint(idx1+2, len(path))
            node1 = traj[path[idx1]]
            node2 = traj[path[idx2]]
            dist = np.linalg.norm(node2 - node1)
            N = int(dist / step_length) + 2
            state_list = self.interpolate(node1, node2, N)
            free = True
            for state in state_list:
                if self.check_collision(state) is True:
                    free = False
                    break

            if free is True:
                idx_remove = slice(idx1+1, idx2)
                del path[idx_remove]
        return path

    def interpolate_traj(self, path, traj, step_length=0.1):
        states = []
        for idx in range(len(path) - 1):
            node1 = traj[path[idx]]
            node2 = traj[path[idx + 1]]
            dist = np.linalg.norm(node2 - node1)
            N = int(dist / step_length) + 2
            path_interpolated = lin_interpolate(node1, node2, N)
            for state in path_interpolated:
                states += [state]
        return states
    


def check_collision(robot_id, object_ids, omit_indices=[-1]):
    col_info = []
    is_col = False
    for object_id in object_ids:
        ress = p.getClosestPoints(robot_id, object_id, distance=2)
        for res in ress:
            if res[3] in omit_indices:
                continue
            if res[8] < 0:
                is_col = True
                col_info += [(res[3], res[4], res[7], res[8])]  # linkA, linkB, normal, distance
    return is_col, col_info

def lin_interpolate(state1, state2, n=1.):
    state_list = []
    for i in range(n+1):
        state_list.append(state1 + 1.*i*(state2-state1)/n)
    return state_list

class interpolator():
    def __init__(self):
        pass

    def interpolate(self, state1, state2, N):
        states = lin_interpolate(state1, state2, N)
        return states


class sampler():
    """
    General sampler, given the joint limits
    """

    def __init__(self, joint_limits=None):
        self.dof = joint_limits.shape[1]
        self.joint_limits = joint_limits

    def sample(self, N=1):
        samples = np.random.rand(N, self.dof)
        samples = self.joint_limits[0] + samples * (self.joint_limits[1] - self.joint_limits[0])
        return samples

class col_checker():
    def __init__(self, robot_id, joint_indices, object_ids, omit_indices=[-1], floating_base = False):
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.object_ids = object_ids
        self.omit_indices = omit_indices
        self.floating_base = floating_base

    def check_collision(self, q):
        if self.floating_base:
            set_q(self.robot_id, self.joint_indices, q, True)
        else:
            set_q(self.robot_id, self.joint_indices, q)
        return check_collision(self.robot_id, self.object_ids, omit_indices=self.omit_indices)[0]