from cmath import inf
import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)
        self.last_battle_end = 0
        self.episode_threshold = 50
        self.episodes_since_guide_window_reduction = 0
        self.jsrl_guide_windows = {}
        self.explore_policy_active = False
        self.multi_player = True
        self.num_enemies = config["num_enemies"]
        self.all_args.reward_speed = 1
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        self.opp_policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)
        
        self.opponent_dir = config["all_args"].opponent_dir
        
        # load opponent model
        print("restoring model.......")
        print(str(self.opponent_dir))
        print('=======================================',self.opponent_dir)
        policy_actor_state_dict = torch.load(self.opponent_dir  + 'actor.pt')
        self.opp_policy.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = torch.load(self.opponent_dir  + 'critic.pt')
        self.opp_policy.critic.load_state_dict(policy_critic_state_dict)

        for i in range(self.n_rollout_threads):
            # [guide_window, last_battle]
            self.jsrl_guide_windows[i] = [inf, 0]

    def run(self):
        self.warmup()
        
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)
        
        self.guide_index = 0

        for episode in range(episodes):
            # if self.all_args.jump_start_model_pool_dir:
            #     self.jump_start_policy = self.jump_start_policy_pool[self.guide_index]
            #     # cycle through guide windows.
            #     if self.guide_index < 4:
            #         self.guide_index += 1
            #     else:
            #         self.guide_index = 0

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            if self.multi_player:
                self.available_enemy_actions = np.ones((self.n_rollout_threads, self.num_enemies, 6+self.num_agents), dtype=np.float32)
                self.enemy_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                self.enemy_obs = np.zeros((self.n_rollout_threads, self.num_enemies, 204), dtype=np.float32)

            for step in range(self.episode_length):
                # Sample actions
                # if self.all_args.jump_start_model_dir or self.all_args.jump_start_model_pool_dir:
                #     values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect_guide(step)
                #     explore_values, explore_actions, explore_action_log_probs, explore_rnn_states, explore_rnn_states_critic = self.collect(step)
                # else:
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # for i in range(self.n_rollout_threads):
                #     if (self.all_args.jump_start_model_dir or self.all_args.jump_start_model_pool_dir) and (step - self.jsrl_guide_windows[i][1] >= self.jsrl_guide_windows[i][0]):
                #         values[i] = explore_values[i]
                #         actions[i] = explore_actions[i]
                #         action_log_probs[i] = explore_action_log_probs[i]
                #         rnn_states[i] = explore_rnn_states[i]
                #         rnn_states_critic[i] = explore_rnn_states_critic[i]

                # Obser reward and next obs
                enemy_actions = None
                if self.multi_player:
                    enemy_actions, enemy_rnn_states = self.collect_enemy(self.enemy_obs,self.enemy_rnn_states,self.buffer.masks[step],self.available_enemy_actions)
                obs, enemy_obs, agent_state, enemy_state, rewards, dones, infos, available_actions, available_enemy_actions = self.envs.step(actions,enemy_actions) if self.multi_player else self.envs.step(actions)
                if self.multi_player:
                    self.available_enemy_actions = available_enemy_actions.copy()
                    self.enemy_rnn_states = enemy_rnn_states.copy()
                    self.enemy_obs = enemy_obs.copy()

                # if the policy is swapped to explore, and the sum rewards are greater, then great, move on to the next guide_window, I think? This is one option.
                # if self.guide_window > 0 and self.explore_policy_active and rewards[0][0][0] > 10:
                #    self.guide_window = self.guide_window - 1
                #    self.episodes_since_guide_window_reduction = 0 # set the threshold to 0.
                #    self.guide_policy_last_rewards[0][0][0] = inf

                for index, done in enumerate(dones):
                    if done.all():
                        if self.all_args.reward_speed and infos[index][0]['won']:
                            max_reward = 1840 # Hardcoded from env
                            scale_rate = 20 # Hardcoded from env.
                            map_length = step - self.jsrl_guide_windows[index][1]
                            upper_map_length_limit = 150
                            norm_max = 30
                            z1 = (map_length / upper_map_length_limit) * norm_max
                            z1 = (z1 - norm_max) * -1 # invert reward to reward lower map_length
                            speed_reward = z1 / (max_reward / scale_rate)
                            rewards[index] += speed_reward

                        self.jsrl_guide_windows[index][1] = step

                        # If we haven't set the guide window yet, set it to the last frame of last attempt, that's our starting point.
                        if self.jsrl_guide_windows[index][0] is inf:
                            self.jsrl_guide_windows[index][0] = step


                data = obs, enemy_obs, agent_state, enemy_state, rewards, dones, infos, available_actions, available_enemy_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 


                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []                    

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    if self.use_wandb:
                        wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                        wandb.log({"guide_window": self.jsrl_guide_windows[0][0]}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                
                self.log_train(train_infos, total_num_steps)

            # If we make it through some number of episodes, just adjust guide window anyway
            # if (self.all_args.jump_start_model_dir or self.all_args.jump_start_model_pool_dir) and self.episodes_since_guide_window_reduction >= self.episode_threshold:
            #     for key in self.jsrl_guide_windows.keys():
            #         if self.jsrl_guide_windows[key][0] > 0:
            #             self.jsrl_guide_windows[key][0] = self.jsrl_guide_windows[key][0] - 1

                self.episodes_since_guide_window_reduction = -1

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
                #self.eval_envs.envs[0].save_replay()

            self.episodes_since_guide_window_reduction += 1

    def warmup(self):
        # reset env
        obs, opp_obs, agent_state, enemy_state, available_actions, avail_enemy_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            agent_state = obs

        self.buffer.share_obs[0] = agent_state.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()
        self.enemy_obs = opp_obs


    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    # @torch.no_grad()
    # def collect_guide(self, step):
    #     self.trainer.prep_rollout()
    #     value, action, action_log_prob, rnn_state, rnn_state_critic \
    #         = self.trainer.jump_start_policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
    #                                         np.concatenate(self.buffer.obs[step]),
    #                                         np.concatenate(self.buffer.rnn_states[step]),
    #                                         np.concatenate(self.buffer.rnn_states_critic[step]),
    #                                         np.concatenate(self.buffer.masks[step]),
    #                                         np.concatenate(self.buffer.available_actions[step]))
    #     # [self.envs, agents, dim]
    #     values = np.array(np.split(_t2n(value), self.n_rollout_threads))
    #     actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
    #     action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
    #     rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
    #     rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

    #     return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    @torch.no_grad()
    def collect_enemy(self, obs, prev_rnn_states, masks, available_actions):
        # self.enemy_trainer.prep_rollout()
        # print(len(obs[0][0]), len(prev_rnn_states[0][0]), len(masks[0][0]), len(available_actions[0][0]))
        actions, rnn_states = \
            self.opp_policy.act(np.concatenate(obs),
                                    np.concatenate(prev_rnn_states),
                                    np.concatenate(masks),
                                    np.concatenate(available_actions),
                                    deterministic=True)
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

        return actions, rnn_states

    def insert(self, data):
        obs, opp_obs, agent_state, enemy_state, rewards, dones, infos, available_actions, enemy_avail_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            agent_state = obs

        self.buffer.insert(agent_state, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        info_we_want_to_keep = ['average_step_rewards', 'dead_ratio']
        for k, v in train_infos.items():
            if k not in info_we_want_to_keep:
                continue # Skip info we don't care about.

            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []
        eval_episode_healths = []

        eval_obs, opp_obs, eval_agent_state, eval_opp_state, eval_available_actions, opp_avail_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        
        opp_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        opp_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        eval_env_infos = {'total_health_remaining': 0, 'eval_average_episode_rewards': 0}

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            opp_actions, opp_rnn_states = self.collect_enemy(self.enemy_obs,opp_rnn_states,opp_masks,opp_avail_actions)
            
            
            
            # Obser reward and next obs
            previous_state = eval_agent_state
            eval_obs, opp_obs, eval_agent_state, eval_opp_state, eval_rewards, eval_dones, \
            eval_infos, eval_available_actions, eval_available_enemy_actions = self.eval_envs.step(eval_actions, opp_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            # Get relative health and shield values for units, this will only work with protoss?
            featureCount = 22

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1

                    total_relative_health = 0
                    total_relative_shield = 0
                    for agent in range(self.num_agents):
                        healthIdx = agent * featureCount
                        shieldIdx = healthIdx + 4
                        total_relative_health += previous_state[eval_i][agent][healthIdx]
                        total_relative_shield += previous_state[eval_i][agent][shieldIdx]

                    if eval_infos[eval_i][0]['won']:
                        eval_env_infos['total_health_remaining'] = (total_relative_shield + total_relative_health) / (self.num_agents * 2)
                    else:
                        eval_env_infos['total_health_remaining'] = 0 


                    eval_episode_healths.append(eval_env_infos['total_health_remaining'])
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_episode_healths = np.array(eval_episode_healths)
                eval_env_infos['eval_average_episode_rewards'] = eval_episode_rewards
                eval_env_infos['total_health_remaining'] = eval_episode_healths
        
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))

                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                    wandb.log({"total_health_remaining": eval_episode_healths.mean()}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
