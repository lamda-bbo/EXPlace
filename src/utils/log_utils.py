"""
Log-related helper functions for training metrics, runtime and best metrics CSV.
"""
import os


def save_runtime(path, episode, state_transition_time, training_time):
    """Append one row to runtime.csv. Call when episode_counter % 10 == 0."""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write("episode,state_transition_time,training_time\n")
    with open(path, 'a') as f:
        f.write(f"{episode},{state_transition_time:.4f},{training_time:.4f}\n")


def save_best_metrics(path, episode, metric_type, tns, gp_hpwl, wns, reward_info, actor_loss, critic_loss):
    """Append one row when best TNS or best GP-HPWL is updated. metric_type is 'tns' or 'gp_hpwl'."""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write("episode,metric_type,tns,wns,gp_hpwl,scaled_reward,hpwl,hierarchy_cost,regularity,displacement_cost,port_cost,df_cost,actor_loss,critic_loss\n")
    r = reward_info or {}
    with open(path, 'a') as f:
        f.write(f"{episode},{metric_type},{tns},{wns},{gp_hpwl},{r.get('scaled_reward', '')},{r.get('wire', '')},{r.get('hier', '')},{r.get('reg', '')},{r.get('displacement', '')},{r.get('port', '')},{r.get('df', '')},{actor_loss},{critic_loss}\n")


def save_eval_metrics(path, writer, episode, hpwl, hierarchy_cost, regularity, displacement_cost, port_cost, df_cost, scaled_reward, tns, wns, gp_hpwl, actor_loss, critic_loss):
    """Append one row to eval_metrics.csv and log scalars to TensorBoard writer."""
    if not os.path.exists(path):
        with open(path, 'a') as f:
            f.write(f"episode,scaled_reward,hpwl,hierarchy_cost,regularity,displacement_cost,port_cost,df_cost,tns,wns,gp_hpwl,actor_loss,critic_loss\n")
    with open(path, 'a') as f:
        f.write(f'{episode},{scaled_reward},{hpwl},{hierarchy_cost},{regularity},{displacement_cost},{port_cost},{df_cost},{tns},{wns},{gp_hpwl},{actor_loss},{critic_loss}\n')

    # TensorBoard logging
    writer.add_scalar('loss/total_reward', scaled_reward, episode)
    writer.add_scalar('cost/hpwl', hpwl, episode)
    writer.add_scalar('cost/hier', hierarchy_cost, episode)
    writer.add_scalar('cost/reg', regularity, episode)
    writer.add_scalar('cost/displacement', displacement_cost, episode)
    writer.add_scalar('cost/port', port_cost, episode)
    writer.add_scalar('cost/df', df_cost, episode)
    writer.add_scalar('cost/tns', tns, episode)
    writer.add_scalar('cost/wns', wns, episode)
    if gp_hpwl != 0:
        writer.add_scalar('cost/gp_hpwl', gp_hpwl, episode)
        writer.add_scalar('loss/actor_loss', actor_loss, episode)
        writer.add_scalar('loss/critic_loss', critic_loss, episode)
