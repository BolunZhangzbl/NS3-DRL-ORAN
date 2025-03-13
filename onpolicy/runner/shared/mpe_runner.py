import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import log_gen
import wandb
import imageio
import os
# from log_gen import *

 
# rus = 4
# states = list(it.product([5,10], repeat=rus))rus**len([5,10])
states=[5,10,20,30,40]

def _t2n(x):
    return x.detach().cpu().numpy()gut

class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def run(self):
        self.pmax=60
        self.act=np.array([20,30,40,50,60])
        # if os.path.exists("/home/waleed/holistic/fifo2"):
        #      os.unlink("/home/waleed/holistic/fifo2")

        # os.mkfifo("/home/waleed/holistic/fifo2")
        # fifo=os.open("/home/waleed/holistic/fifo1",os.O_RDONLY) #it frezez here runs after runing sim2
        # fifo2=os.open("/home/waleed/holistic/fifo2", os.O_WRONLY)
        # global last
        # print("Opening FIFO...")

        # last=0  
        self.warmup() 
        if os.path.exists("/home/waleed/holistic/fifo2"):
            os.unlink("/home/waleed/holistic/fifo2")

        os.mkfifo("/home/waleed/holistic/fifo2")
        self.fifo=os.open("/home/waleed/holistic/fifo1",os.O_RDONLY) #it frezez here runs after runing sim2
        self.fifo2=os.open("/home/waleed/holistic/fifo2", os.O_WRONLY)
        last=[]
        print("Opening FIFOs to send/recive...") # run sim2 in ns 3 holistic dir -----------------------
        # w=log_gen.get_data(last,fifo)
        # pwr=w.get("txpower")
        # pwr=[int(item) for item in pwr]
        # a=log_gen.send_action(w.get("txpower"),fifo2)

        # self.obs2_sh=[[[[0.0]*(self.len*self.num_agents)]*self.n_eval_rollout_threads]*self.num_agents]
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions  frim sim2.cc  where carriers are 2 auto selected, numb of enb and ues can be tuned...
                #dimension for available actions=#neurons 25, threads 1, agents 2, total actions 5
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)#def act_dim and act_env
                    
                # Obser reward and next obs

                # obs, rewards, dones, infos = self.envs.step(actions_env)
                obs, rewards, dones, infos = self.step(actions_env)

                print("a1",rewards)
                
                # print("obs",obs[0][:])
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                
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
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                # self.log_train(train_infos, total_num_steps)------------for wandb 
                # self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        #obs = self.envs.reset()#3d size=threads,agent,statespace len
        # fifo=os.open("/home/waleed/holistic/fifo1",os.O_RDONLY) 
        # d=log_gen.get_data(fifo)
        # action_p=np.array(list(map(int,list(d.get('txpower')))))
        self.len=1
        obs=([[[self.pmax]*self.n_rollout_threads]*self.num_agents])
        obs=(np.asarray(obs)).T
        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)#3d size=threads,agent,statespaces of all agents
        else:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        # step  this is  env.step()

    def step(self, action_n):#action_n saves actions for each agent
        powers=np.zeros(self.num_agents)
        p=np.array([self.pmax])
        obs_n=[]
        reward_n =[]
        done_n = []
        info_n = []
        # set action for each agent
        for j in range(self.n_rollout_threads):
            for i in range(self.num_agents):            
                powers[i]=self.act[np.where(action_n[j][i]==1)]               
        # advance world state
        #self.world.step()  # core.step()
        # if os.path.exists("/home/waleed/holistic/fifo2"):
        #     os.unlink("/home/waleed/holistic/fifo2")

        # os.mkfifo("/home/waleed/holistic/fifo2")
        # fifo=os.open("/home/waleed/holistic/fifo1",os.O_RDONLY) #it frezez here runs after runing sim2
        # fifo2=os.open("/home/waleed/holistic/fifo2", os.O_WRONLY)
        # last=[]
        # print("Opening FIFOs to send/recive...") # run sim2 in ns 3 holistic dir -----------------------------------------------
        con = np.concatenate((p,powers,p))
        txp=','.join(con.astype(str))
        print('action taken',txp)
        log_gen.send_action(txp,self.fifo2)
        # record observation for each agent
        
        # fifo=os.open("/home/waleed/holistic/fifo1",os.O_RDONLY) 
        
        
        l=log_gen.last
        d,l=log_gen.get_data(l,self.fifo)
        log_gen.last+=l
        tpp=np.array(list(map(int,list(d.get('tp')))))
        for i in range(self.num_agents):
            obs_n.append(tpp[i])
            reward_n.append([tpp[i]])
            if i<1:
                done_n.append([False])
            else:
                done_n[0].append(False)
            info = {'individual_reward': tpp[i]}
            # env_info = (i)
            # if 'fail' in env_info.keys():
            #     info['fail'] = env_info['fail']
            info_n.append(info)
        done_n=np.asarray(done_n)
        obs_n=np.asarray(obs_n)
        # all agents get total reward in cooperative case, if shared reward, all agents have the same reward, and reward is sum
        reward = np.sum(reward_n)
        # if self.shared_reward:
        reward_n = [[reward]] * self.num_agents

        # if self.post_step_callback is not None:
        #     self.post_step_callback(self.world)

        return obs_n, reward_n, done_n, info_n

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()#init actor and critic n/ws
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))# each step is for each index
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        # #==========================define action_env with shape(n_threads,n_agents,action len)
        
        actions_env = np.squeeze(np.eye(len(self.act))[actions], 2)
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        obs=obs.reshape(self.n_rollout_threads,self.num_agents,1)
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))  
            eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)


            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

   