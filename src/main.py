import numpy as np
import torch as th
import random
import os
import sys
import time
import argparse
import warnings
from datetime import datetime
import yaml
import pdb
import pickle
import shutil
import ray
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))  # src/ directory
DREAMPLACE_PARENT_DIR = os.path.join(ROOT_DIR, "DREAMPlace", "install")
DREAMPLACE_DIR = os.path.join(DREAMPLACE_PARENT_DIR, "dreamplace")
sys.path.extend(
    [SRC_DIR, DREAMPLACE_PARENT_DIR, DREAMPLACE_DIR]
)

from collections import namedtuple
from types import SimpleNamespace
from agent import PPOAgent
from env_utils import init_ray_envs, create_single_env
from problem_instance import ProblemInstance
from utils.log_utils import save_runtime, save_best_metrics, save_eval_metrics

CONFIG_DIR = os.path.join(ROOT_DIR, "config")

Transition = namedtuple('Transition',['state', 'action', 'reward', 'action_log_prob', 'done'])


def process_args():
    """Load config from YAML file and override with command line arguments"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RL Training and Inference for Macro Placement')
    parser.add_argument('--config', type=str, default='or', help='Config file name (without .yaml)')
    parser.add_argument('--benchmark', type=str, required=True, help='Design benchmark name')
    parser.add_argument('--seed', type=int, help='Random seed (overrides config)')
    parser.add_argument('--gpu', type=int, help='GPU device ID (overrides config)')
    parser.add_argument('--test', action='store_true', default=False, help='Enable test mode')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model for test (e.g. best_model_ep1500.pt)')
    parser.add_argument('--visualize', action='store_true', default=False, help='Visualize step-by-step decisions during test (saves to log_dir/visulization_steps/)')
    cmd_args = parser.parse_args()
    
    # Load config from YAML file
    config_path = os.path.join(CONFIG_DIR, f"{cmd_args.config}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    # Override config with command line arguments
    if cmd_args.seed is not None:
        config_dict['seed'] = cmd_args.seed
    if cmd_args.gpu is not None:
        config_dict['gpu'] = cmd_args.gpu
    config_dict['test'] = cmd_args.test
    config_dict['debug'] = cmd_args.debug
    config_dict['benchmark'] = cmd_args.benchmark
    config_dict['config'] = cmd_args.config
    if cmd_args.model_path is not None:
        config_dict['model_path'] = cmd_args.model_path
    if cmd_args.visualize:
        config_dict['visualize_flag'] = True
    
    # Convert to SimpleNamespace
    args = SimpleNamespace(**config_dict)
    
    # Set runtime attributes
    args.design_name = args.benchmark
    print(f"benchmark:\t{args.design_name}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if th.cuda.is_available() and args.use_cuda:
        args.device = 'cuda'
    else:
        args.use_cuda = False
        args.device = 'cpu'
    print(f'using device: {args.device}')
    
    args.unique_token = f"seed_{args.seed}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    assert args.grid % 32 == 0, 'grid should be a multiple of 32'
    
    return args, config_path

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

def main(args, config_path):
    args.i_episode = 0
    seed_torch(args.seed)

    # Log files
    log_name = "rl_logs"
    log_dir = os.path.join(args.log_dir, log_name, "{}".format(args.design_name)) 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.log_dir = os.path.join(log_dir, f"{timestamp}")
    args.model_dir = os.path.join(args.log_dir, "model")
    args.placement_dir = os.path.join(args.log_dir, "placement")
    args.visualization_dir = os.path.join(args.log_dir, "visualization")
    if not args.debug:
        os.makedirs(args.log_dir, exist_ok=True)
        # Save config file
        shutil.copy(config_path, args.log_dir)
        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(args.placement_dir, exist_ok=True)
        os.makedirs(args.visualization_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # Initialize test problem
    test_problem = ProblemInstance(args, args.design_name, init=False)
    # Debug mode
    if args.debug:
        env = create_single_env(args, test_problem)
        agent = PPOAgent(args=args)
        run_single_env(env, agent, args, reward_scaling_flag=True)
        run_single_env(env, agent, args, reward_scaling_flag=False)
        pdb.set_trace()

    # Initialize ray environments - each worker creates its own environment
    print(f"Initializing Ray environments, number of processes: {args.rollout_batch_size}")
    ray_envs = init_ray_envs(args, args.rollout_batch_size, test_problem)
    test_env = create_single_env(args, test_problem)

    # Initialize environment and agent
    agent = PPOAgent(args=args)
    agent.train()
    seed_torch(args.seed)

    # Reward scaling
    if args.use_reward_scaling:
        _, _ = run(ray_envs, agent, args, reward_scaling_flag=True)

    # Visualize prototype
    ray.get(ray_envs[0].visualize_prototype.remote())

    # Training loop
    num_batches = args.episode // args.rollout_batch_size
    t_0 = time.time()
    best_reward = float('-inf')
    best_reward_info = None
    episode_counter = 0
    best_tns = float('-inf')
    best_gp_hpwl = float('inf')  # lower is better
    best_metrics_path = os.path.join(args.log_dir, 'best_metrics.csv')
    runtime_path = os.path.join(args.log_dir, 'runtime.csv')

    for batch_idx in range(1, num_batches + 1):
        episode_counter += args.rollout_batch_size
        t_start = time.time()
        reward_info, _ = run(ray_envs, agent, args)
        t_rollout = time.time() - t_start
        t_update_start = time.time()
        _, actor_loss, critic_loss = agent.update()
        t_train = time.time() - t_update_start

        # Record runtime every 10 episodes
        if episode_counter % 10 == 0 and not args.debug:
            save_runtime(runtime_path, episode_counter, t_rollout, t_train)

        # Evaluate with DMP periodically
        gp_hpwl, tns, wns = 0., 0., 0.
        if episode_counter % 100 == 0:
            if args.use_dmp_for_evaluation:
                trajectory = inference(args, model_path=None, agent=agent, test_env=test_env)
                gp_hpwl, tns, wns = test_problem.evaluate(trajectory['macro_pos'])
                # Update best TNS (higher is better): save macro_pos, .def, .png and model
                if tns > best_tns:
                    best_tns = tns
                    th.save(trajectory['macro_pos'], os.path.join(args.placement_dir, 'best_tns_macro_pos.pt'))
                    test_problem.save_placement(os.path.join(args.placement_dir, 'best_tns.def'))
                    test_problem.plot(gp_hpwl, os.path.join(args.visualization_dir, 'best_tns.png'))
                    agent.save_model(episode=episode_counter, path=args.model_dir, filename='best_tns_model.pt')
                    save_best_metrics(best_metrics_path, episode_counter, 'tns', tns, gp_hpwl, wns, reward_info, actor_loss, critic_loss)
                    print(f"Best TNS updated - TNS: {tns}, WNS: {wns}, GP-HPWL: {gp_hpwl}")
                # Update best GP-HPWL (lower is better): save macro_pos, .def, .png and model
                if gp_hpwl < best_gp_hpwl:
                    best_gp_hpwl = gp_hpwl
                    th.save(trajectory['macro_pos'], os.path.join(args.placement_dir, 'best_gp_hpwl_macro_pos.pt'))
                    test_problem.save_placement(os.path.join(args.placement_dir, 'best_gp_hpwl.def'))
                    test_problem.plot(gp_hpwl, os.path.join(args.visualization_dir, 'best_gp_hpwl.png'))
                    agent.save_model(episode=episode_counter, path=args.model_dir, filename='best_gp_hpwl_model.pt')
                    save_best_metrics(best_metrics_path, episode_counter, 'gp_hpwl', tns, gp_hpwl, wns, reward_info, actor_loss, critic_loss)
                    print(f"Best GP-HPWL updated - GP-HPWL: {gp_hpwl}, TNS: {tns}, WNS: {wns}")
                agent.train()

        if not args.debug:
            save_eval_metrics(
                path=os.path.join(args.log_dir, 'eval_metrics.csv'),
                writer=writer,
                episode=episode_counter,
                hpwl=reward_info.get('wire', 0),
                hierarchy_cost=reward_info.get('hier', 0),
                regularity=reward_info.get('reg', 0),
                displacement_cost=reward_info.get('displacement', 0),
                port_cost=reward_info.get('port', 0),
                df_cost=reward_info.get('df', 0),
                scaled_reward=reward_info.get('scaled_reward', 0),
                tns=tns,
                wns=wns,
                gp_hpwl=gp_hpwl,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
            )

        # Switch to corner at late stage
        if not args.corner_flag and episode_counter + args.late_stage_episode >= args.episode:
            args.corner_flag = True
            print(f"Switch to corner at late stage")

        # Keep the best total_reward info across all batches
        current_reward = reward_info.get('scaled_reward', float('-inf'))
        if current_reward > best_reward:
            best_reward = current_reward
            best_reward_info = reward_info.copy()

        # Current time step model checkpoint (single file, overwrite each batch)
        if not args.debug:
            agent.save_model(episode=episode_counter, path=args.model_dir, filename='current_model.pt')
        # Visualize placement for the best env in this batch -> save to visualization dir (via args.visualization_dir in env)
        ray.get(ray_envs[reward_info['batch_best_idx']].visualize_placement.remote(episode_counter))

        # Print best episode info
        best_info_no_macro = best_reward_info.copy()
        if 'macro_pos' in best_info_no_macro:
            del best_info_no_macro['macro_pos']
        print(f"Batch {batch_idx}/{num_batches} (Episode {episode_counter}) - Time: {time.time() - t_start:.2f}s")
        print(f"Best scaled_reward: {best_reward}")
        print(f"Best episode info: {best_info_no_macro}")

    # Total runtime
    print(f"Training completed. Time: {time.time() - t_0:.2f}s")
    writer.close()  # Close TensorBoard writer
    
    # Clean up Ray resources
    ray.shutdown()
    return

def run(ray_envs, agent, args, reward_scaling_flag=False):
    """
    Run one episode in parallel environments, return the reward_info with the best total_reward among all envs.
    Returns:
        reward_info: dict, info for this episode (best total_reward among all envs), with 'batch_best_idx' field
    """
    # Reset all environments in parallel
    visualize_flag = getattr(args, 'visualize_flag', False)
    states = ray.get([env.reset.remote(reward_scaling_flag, args.corner_flag, visualize_flag) for env in ray_envs])
    dones = False
    episode_infos = [None] * args.rollout_batch_size
    episode_rewards = [float('-inf')] * args.rollout_batch_size
    
    # Collect all transitions for each environment
    env_transitions = [[] for _ in range(args.rollout_batch_size)]
    
    while not dones:
        # Batch states for agent
        batch_state = np.concatenate(states, axis=0)
        # Agent selects actions
        actions, action_log_probs = agent.select_action(batch_state)
        # Step all environments in parallel
        results = ray.get([env.step.remote(a) for env, a in zip(ray_envs, actions)])
        next_states, rewards, dones_list, infos = zip(*results)
        
        # Collect transitions for each environment (don't store yet)
        for idx in range(args.rollout_batch_size):
            trans = Transition(state=states[idx].copy(),  # Use copy to avoid reference issues
                               action=actions[idx],
                               reward=th.tensor(rewards[idx]) / 200.0,
                               action_log_prob=action_log_probs[idx],
                               done=dones_list[idx])
            env_transitions[idx].append(trans)
            
            if dones_list[idx]:
                episode_infos[idx] = infos[idx]
                episode_rewards[idx] = infos[idx].get('scaled_reward', float('-inf'))
        
        states = next_states
        dones = dones_list[0]
    
    # Store transitions in episode order: all steps from env0, then all stepsps from env1, etc.
    for env_idx in range(args.rollout_batch_size):
        for trans in env_transitions[env_idx]:
            agent.store_transition(trans)
    
    # Find the best reward_info in this batch
    best_idx = int(np.argmax(episode_rewards))
    best_reward_info = episode_infos[best_idx]
    if best_reward_info is not None:
        best_reward_info = best_reward_info.copy()
        best_reward_info['batch_best_idx'] = best_idx
    return best_reward_info, episode_infos

def run_single_env(env, agent, args, reward_scaling_flag=False):
    """
    Run one episode in a single environment, return the reward_info with the best total_reward.
    """
    visualize_flag = getattr(args, 'visualize_flag', False)
    state = env.reset(reward_scaling_flag=reward_scaling_flag, visualize_flag=visualize_flag)
    done = False
    while not done:
        state = np.concatenate([state], axis=0)
        action, _ = agent.select_action(state)
        next_state, reward, done, info = env.step(action[0])
        state = next_state
        done = done or info.get('done', False)
    return info


def inference(args, model_path=None, agent=None, test_env=None):
    """
    Load trained model and perform inference
    
    Args:
        args: Configuration parameters
        model_path (str): Model file path, usually 'best_model.pt'
        agent: Optional pre-initialized agent
        test_env: Optional pre-initialized environment
    
    Returns:
        dict: Information of the inference result, including reward, macro_pos, etc.
    """
    if model_path is not None:
        print(f"Starting model test: {model_path}")
        args.log_dir = os.path.dirname(model_path)
        # Initialize agent and load model
        agent = PPOAgent(args=args)
        agent.load_model(model_path)
        test_problem = ProblemInstance(args, args.benchmark, init=False)
    else:
        test_problem = None

    agent.eval()  # Set to evaluation mode

    if test_env is None:
        # Create single environment
        env = create_single_env(args, test_problem)
    else:
        env = test_env
        args.use_reward_scaling = False

    # Run single inference
    reward_info = run_single_env(env, agent, args)
    
    if reward_info is not None:
        current_reward = reward_info.get('scaled_reward', 0)
        print(f"Inference reward: {current_reward:.4f}")
        
        if model_path is not None and test_problem is not None:
            gp_hpwl, tns, wns = test_problem.evaluate(reward_info['macro_pos'])
            test_problem.save_placement(os.path.join(args.log_dir, 'test_dmp.def'))
            test_problem.plot(gp_hpwl, os.path.join(args.log_dir, 'test_dmp.png'))
            print(f"Timing results - GP_HPWL: {gp_hpwl}, TNS: {tns}, WNS: {wns}")
            # log timing results
            log_file = "rl_logs/test_results.csv"
            with open(log_file, 'a') as f:
                f.write(f"{args.benchmark}, {gp_hpwl}, {tns}, {wns}\n")
        
        env.visualize_placement(10000, test_mode=True)
    
    args.use_reward_scaling = True
    return reward_info

if __name__ == '__main__':
    args, config_path = process_args()
    args.model_path = getattr(args, 'model_path', None)
    # Check if there are test parameters
    if args.test:
        if args.model_path is None or not os.path.isfile(args.model_path):
            print("Test mode requires a valid --model_path (e.g. rl_logs/superblue3/<run>/model/best_model_ep1500.pt)")
            sys.exit(1)
        print(f"Test parameters:")
        print(f"  Model path: {args.model_path}")
        print(f"  Design name: {args.benchmark}")
        if getattr(args, 'visualize_flag', False):
            print(f"  Step visualization: enabled -> log_dir/visulization_steps/step_*.jpg")
        result = inference(args=args, model_path=args.model_path)
    else:
        # Normal training mode
        main(args, config_path)
    
