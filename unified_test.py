from time import perf_counter  # Replace clock with perf_counter
from acktr.model_loader import nnModel
from acktr.reorder import ReorderTree
import gym
import copy
from gym.envs.registration import register
from acktr.arguments import get_args
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.animation as animation
import time

def update_3d_packing(env, ax):
    """
    Update the 3D visualization
    """
    ax.cla()  # Clear the current axes
    
    # Draw container edges
    container_size = env.bin_size
    for i in [0, container_size[0]]:
        for j in [0, container_size[1]]:
            ax.plot([i,i], [j,j], [0,container_size[2]], 'k-', alpha=0.3)
            ax.plot([i,i], [0,container_size[1]], [j,j], 'k-', alpha=0.3)
            ax.plot([0,container_size[0]], [i,i], [j,j], 'k-', alpha=0.3)

    # Draw each box
    for box in env.space.boxes:
        x, y, z = box.lx, box.ly, box.lz
        dx, dy, dz = box.x, box.y, box.z
        
        vertices = np.array([
            [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
            [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
        ])
        
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[4], vertices[7], vertices[3], vertices[0]]
        ]
        
        poly3d = Poly3DCollection(faces, alpha=0.4)
        poly3d.set_facecolor(np.random.rand(3))
        poly3d.set_edgecolor('black')
        ax.add_collection3d(poly3d)

    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Bin Packing Animation')
    ax.set_xlim(0, container_size[0])
    ax.set_ylim(0, container_size[1])
    ax.set_zlim(0, container_size[2])
    ax.view_init(elev=30, azim=45)

def run_sequence(nmodel, raw_env, preview_num, c_bound):
    env = copy.deepcopy(raw_env)
    obs = env.cur_observation
    default_counter = 0
    box_counter = 0
    start = perf_counter()
    
    # Setup the figure for animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()  # Turn on interactive mode
    
    print("\nPacking sequence:")
    print("Format: Box #{box_num}: size({x},{y},{z}) at position({lx},{ly},{lz})")
    
    while True:
        # Step 1: Preview 
        box_list = env.box_creator.preview(preview_num)

        # Step 2: Reorder
        tree = ReorderTree(nmodel, box_list, env, times=100)
        act, val, default = tree.reorder_search()

        # Step 3: Step
        obs, _, done, info = env.step([act])
        
        # Print and visualize current box
        if len(env.space.boxes) > box_counter:
            current_box = env.space.boxes[-1]
            print(f"Box #{box_counter + 1}: size({current_box.x},{current_box.y},{current_box.z}) "
                  f"at position({current_box.lx},{current_box.ly},{current_box.lz})")
            
            # Update visualization
            update_3d_packing(env, ax)
            plt.draw()
            plt.pause(0.5)  # Pause to show the animation
        
        # Step 4: Check done
        if done:
            end = perf_counter()
            print('\nFinal Results:')
            print('Time cost:', end-start)
            print('Space utilization ratio:', info['ratio'])
            print(f'Total boxes placed: {info["counter"]}')
            
            # Show final state
            plt.ioff()  # Turn off interactive mode
            plt.show()
            
            return info['ratio'], info['counter'], end-start, default_counter/box_counter
            
        box_counter += 1
        default_counter += int(default)

def unified_test(url,  args, pruning_threshold = 0.5):
    nmodel = nnModel(url, args)
    data_url = './dataset/' +args.data_name
    env = gym.make(args.env_name,
                    box_set=args.box_size_set,
                    container_size=args.container_size,
                    test=True, data_name=data_url,
                    enable_rotation=args.enable_rotation,
                    data_type=args.data_type)
    print('Env name: ', args.env_name)
    print('Data url: ', data_url)
    print('Model url: ', url)
    print('Case number: ', args.cases)
    print('pruning threshold: ', pruning_threshold)
    print('Known item number: ', args.preview)
    times = args.cases
    ratios = []
    avg_ratio, avg_counter, avg_time, avg_drate = 0.0, 0.0, 0.0, 0.0
    c_bound = pruning_threshold
    for i in range(times):
        if i % 10 == 0:
            print('case', i+1)
        env.reset()
        env.box_creator.preview(500)
        ratio, counter, time, depen_rate = run_sequence(nmodel, env, args.preview, c_bound)
        avg_ratio += ratio
        ratios.append(ratio)
        avg_counter += counter
        avg_time += time
        avg_drate += depen_rate

    print()
    print('All cases have been done!')
    print('----------------------------------------------')
    print('average space utilization: %.4f'%(avg_ratio/times))
    print('average put item number: %.4f'%(avg_counter/times))
    print('average sequence time: %.4f'%(avg_time/times))
    print('average time per item: %.4f'%(avg_time/avg_counter))
    print('----------------------------------------------')

def registration_envs():
    register(
        id='Bpp-v0',                                  # Format should be xxx-v0, xxx-v1
        entry_point='envs.bpp0:PackingGame',   # Expalined in envs/__init__.py
    )

if __name__ == '__main__':
    registration_envs()
    args = get_args()
    pruning_threshold = 0.5  # pruning_threshold (default: 0.5)
    unified_test('pretrained_models/default_cut_2.pt', args, pruning_threshold)
    # args.enable_rotation = True
    # unified_test('pretrained_models/rotation_cut_2.pt', args, pruning_threshold)

    
