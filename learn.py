import numpy as np
import sys
import torch
import torch.nn as nn
from utils.replay_buffer import ReplayMemory, Transition
from utils.general import export_plot, CSVLogger
from collections import deque

class DQNTrainer(object):
    def __init__(self,
                 env,
                 exploration_schedule,
                 lr_schedule,
                 config,
                 q_function
                 ):
        self.env = env
        self.state_shape = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.config = config
        self.q_network = q_function(self.state_shape,
                                    self.num_actions)
        self.target_network = q_function(self.state_shape,
                                         self.num_actions)

        def init_weights(m):
            if hasattr(m, 'weight'):
                nn.init.orthogonal_(m.weight.data)
            if hasattr(m, 'bias'):
                nn.init.constant_(m.bias.data, 0)

        self.q_network.apply(init_weights)
        self.target_network.load_state_dict(
            self.q_network.state_dict())
        self.exploration_schedule = exploration_schedule
        self.lr_schedule = lr_schedule
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), betas=(0.9, 0.999), eps=1e-8)

    @staticmethod
    def _extract_state(reset_output):
        if isinstance(reset_output, tuple):
            return reset_output[0]
        return reset_output

    @staticmethod
    def _extract_step(step_output):
        if len(step_output) == 5:
            next_state, reward, terminated, truncated, info = step_output
            done = terminated or truncated
            return next_state, reward, done, info
        return step_output

    def learn(self):
        '''

        :param env:
        :param q_fun:
        :param exploration_schedule:
        :param lr_schedule:
        :param q_optimizer:
        :param config:
        :return:
        '''
        replay_buffer = ReplayMemory(self.config.replay_buffer_size)
        data_dict = {
            'Timestep': 0,
            'Training Rewards': 0,
            'Max Q': 0,
            'Eval Rewards': 0,
            'Loss': 0
        }
        fieldnames = [key for key, _ in data_dict.items()]
        csv_logger = CSVLogger(fieldnames=fieldnames,
                               filename=self.config.csv_dir)
        train_reward_window = getattr(self.config,
                          'train_reward_window',
                          self.config.num_episodes_eval)
        rewards = deque(maxlen=train_reward_window)
        max_q_values = deque(maxlen=self.config.num_episodes_eval)
        loss_val = params_norm = float('inf')
        loss_ema = None
        t = last_eval = 0
        scores_eval = [self.evaluate()]
        train_rewards = []
        num_episodes = 0
        while t < self.config.num_timesteps:
            num_episodes += 1
            state = self._extract_state(self.env.reset())
            done = False
            episode_reward = 0
            while not done and (t < self.config.num_timesteps):
                t += 1
                last_eval += 1
                # choose action according to current Q and exploration
                with torch.no_grad():
                    q_vals = self.q_network(torch.FloatTensor(state))
                    action = self.exploration_schedule.get_action(
                        q_vals.numpy())
                if t > self.config.learning_start:
                    # start schedule on exploration parameter,
                    # otherwise will pick random action
                    # best_action, q_vals = get_best_action(state, t)
                    self.exploration_schedule.update(t -
                                                     self.config.learning_start)
                max_q_values.append(max(q_vals))
                next_state, reward, done, info = self._extract_step(
                    self.env.step(action))

                replay_buffer.push(torch.tensor(state),
                                   torch.tensor(action).unsqueeze(0),
                                   torch.tensor(next_state),
                                   torch.tensor(reward).unsqueeze(0),
                                   torch.tensor(done,
                                                dtype=bool).unsqueeze(
                                       0))
                state = next_state

                # TRAIN
                if (t > self.config.learning_start) and \
                        (len(replay_buffer) >=
                         self.config.batch_size) and (t %
                                                      self.config.learning_freq == 0):
                    loss_val, params_norm = self.training_step(t,
                                                               replay_buffer,
                                                               self.lr_schedule.curr_val)
                    if loss_ema is None:
                        loss_ema = loss_val
                    else:
                        loss_ema = 0.95 * loss_ema + 0.05 * loss_val
                    self.lr_schedule.update(t)

                # LOGGING
                if (t % self.config.log_freq == 0) and (
                        t > self.config.learning_start):
                    if len(rewards) > 0:
                        print(
                            f'Timestep {t - self.config.learning_start} | '
                            f'Episode {num_episodes} | '
                            f'Mean Ep R '
                            f'{np.mean(rewards):.4f} | '
                            f'Max R {np.max(rewards):.4f} | '
                            f'Max Q {np.mean(max_q_values):.4f} | '
                            f'Params Norm {params_norm:.4f} | '
                            f'Loss {(loss_ema if loss_ema is not None else loss_val):.4f} | '
                            f'lr {self.lr_schedule.curr_val:.6f} | '
                            f'eps {self.exploration_schedule.curr_val:.3f}')
                        sys.stdout.flush()
                episode_reward += reward

                if done or t >= self.config.num_timesteps:
                    break

            rewards.append(episode_reward)
            if (t > self.config.learning_start) and (last_eval >=
                                                     self.config.eval_freq):
                last_eval = 0
                train_rewards += [np.mean(rewards)]
                export_plot(train_rewards, "Episode Rewards",
                            self.config.train_plot_dir)
                scores_eval += [self.evaluate()]
                data_dict = {
                    'Timestep': t,
                    'Training Rewards': float(np.mean(rewards)),
                    'Max Q': np.mean(max_q_values),
                    'Eval Rewards': float(scores_eval[-1]),
                    'Loss': float(loss_ema if loss_ema is not None else loss_val)
                }
                csv_logger.writerow(data_dict)
                export_plot(scores_eval, "Episode Rewards",
                            self.config.plot_dir)

        print('Training Finished')
        torch.save(self.q_network.state_dict(),
                   self.config.model_dir)
        scores_eval += [self.evaluate()]
        train_rewards += [np.mean(rewards)]
        export_plot(train_rewards, "Episode Rewards",
                    self.config.train_plot_dir)
        export_plot(scores_eval, "Episode Rewards",
                    self.config.plot_dir)
        csv_logger.close()

    def training_step(self, t, replay_buffer, lr):
        '''

        :param t:
        :param replay_buffer:
        :param lr:
        :return:
        '''
        transitions = replay_buffer.sample(self.config.batch_size)
        batch = Transition(*zip(*transitions))
        done_batch = torch.cat(batch.done).view(
            self.config.batch_size, -1).squeeze()
        next_state_batch = torch.cat(batch.next_state).float().view(
            self.config.batch_size, -1)
        state_batch = torch.cat(batch.state).float().view(
            self.config.batch_size, -1)
        action_batch = torch.cat(batch.action).long().view(
            self.config.batch_size, -1)
        reward_batch = torch.cat(batch.reward).view(
            self.config.batch_size, -1)

        if self.config.double == True:
            loss = self.compute_DoubleDQN_loss(state_batch,
                                               action_batch,
                                               reward_batch,
                                               next_state_batch,
                                               done_batch)
        else:
            loss = self.compute_DQN_loss(state_batch, action_batch,
                                         reward_batch,
                                         next_state_batch, done_batch)
        self.q_optimizer.zero_grad()
        loss.backward()
        total_param_norm = torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.clip_val)

        # set optimizer learning rate
        for group in self.q_optimizer.param_groups:
            group['lr'] = lr
        self.q_optimizer.step()

        # update target network
        tau = getattr(self.config, 'tau', None)
        use_soft_target = isinstance(tau, (float, int)) and (0.0 < float(tau) < 1.0)
        if use_soft_target:
            self.update_target()
        elif (t // self.config.learning_freq) % self.config.target_update_freq == 0:
            self.update_target()

        if t % self.config.saving_freq == 0:
            torch.save(self.q_network.state_dict(),
                       self.config.model_dir)
        return loss.item(), total_param_norm


    def update_target(self):
        '''
        TODO:
        update_target will be called periodically
        to copy Q network weights to target Q network

        In DQN, we maintain two identical Q networks with
        2 different sets of weights.

        Periodically, we need to update all the weights of the Q network
        and assign them with the values from the regular network.

        Hint:
            1. look up loading pytorch models
        '''

        ##############################################################
        ################### YOUR CODE HERE - 1-2 lines ###############

        tau = getattr(self.config, 'tau', None)
        if isinstance(tau, (float, int)) and (0.0 < float(tau) < 1.0):
            tau = float(tau)
            for target_param, online_param in zip(
                    self.target_network.parameters(),
                    self.q_network.parameters()):
                target_param.data.copy_(
                    (1.0 - tau) * target_param.data + tau * online_param.data)
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())

        ##############################################################
        ######################## END YOUR CODE #######################


    def compute_DQN_loss(self, state_batch, action_batch,
                         reward_batch, next_state_batch, done_batch):
        '''
        :param state_batch: (torch tensor) shape = (batch_size x state_dims),
                The batched tensor of states collected during
                training (i.e. s)
        :param action_batch: (torch LongTensor) shape = (batch_size,)
                The actions that you actually took at each step (i.e. a)
        :param reward_batch: (torch tensor) shape = (batch_size,)
                The rewards that you actually got at each step (i.e. r)
        :param (torch tensor) shape = (batch_size x state_dims),
                The batched tensor of next states collected during
                training (i.e. s')
        :param done_batch: (torch tensor) shape = (batch_size,)
                A boolean mask of examples where we reached the terminal state
        :return: loss: (torch tensor) shape = (1)
        '''
        '''
        TODO:               
        Calculate the MSE loss of this step.
        The loss for an example is defined as:
            Q_samp(s) = r if done
                        = r + gamma * max_a' Q_target(s', a')
            loss = (Q_samp(s) - Q(s, a))^2

        You can get the Q values by calling self.q_network and the 
        target values by calling self.target_network
        
        Recall that there should not be any gradients passed 
        through the target network (Hint: you can use "with 
        torch.no_grad()")
        '''
        loss = None
        ##############################################################
        ################ YOUR CODE HERE - 5-6 lines ##################

        q_values = self.q_network(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1, keepdim=True)[0]
            done_mask = done_batch.view(-1, 1).float()
            q_targets = reward_batch + self.config.gamma * next_q_values * (1.0 - done_mask)
            q_clip = getattr(self.config, 'target_q_clip', None)
            if isinstance(q_clip, (float, int)) and q_clip > 0:
                q_targets = torch.clamp(q_targets, -float(q_clip), float(q_clip))
        loss = nn.functional.smooth_l1_loss(q_values, q_targets)
        if getattr(self.config, 'normalize_loss', False):
            loss_scale = q_targets.detach().abs().mean().clamp_min(1.0)
            loss = loss / loss_scale

        ##############################################################
        ######################## END YOUR CODE #######################
        return loss


    def compute_DoubleDQN_loss(self, state_batch, action_batch,
                               reward_batch, next_state_batch,
                               done_batch):
        '''
        :param state_batch: Tensor (batch_size x state_dims),
        batched tensor of states collected during training
        :param action_batch: LongTensor (batch_size x action_dims)
        :param reward_batch: Tensor (batch_size x 1)
        :param next_state_batch: Tensor (batch_size x state_dims)
        batched tensor of next states
        :param done_batch: Tensor of bools (batch_size x 1)
        :return: scalar
        '''
        '''
        TODO:               
        Calculate the MSE loss of this step.
        The loss for an example is defined as:
            Q_samp(s) = r if done
                      = r + gamma * Q_target(s', argmax_{a'} Q(s', a'))
            loss = (Q_samp(s) - Q(s, a))^2
            
        Recall that there should not be any gradients passed 
        through the target network (Hint: you can use "with 
        torch.no_grad()")
        '''
        loss = None
        ##############################################################
        ############### YOUR CODE HERE - 8-10 lines ##################
        q_values = self.q_network(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_actions = self.q_network(next_state_batch).argmax(
                dim=1, keepdim=True)
            next_q_values = self.target_network(next_state_batch).gather(
                1, next_actions)
            done_mask = done_batch.view(-1, 1).float()
            q_targets = reward_batch + self.config.gamma * next_q_values * (
                1.0 - done_mask)
            q_clip = getattr(self.config, 'target_q_clip', None)
            if isinstance(q_clip, (float, int)) and q_clip > 0:
                q_targets = torch.clamp(q_targets, -float(q_clip), float(q_clip))
        loss = nn.functional.smooth_l1_loss(q_values, q_targets)
        if getattr(self.config, 'normalize_loss', False):
            loss_scale = q_targets.detach().abs().mean().clamp_min(1.0)
            loss = loss / loss_scale

        ##############################################################
        ######################## END YOUR CODE #######################
        return loss


    def evaluate(self):
        rewards = []
        for ep in range(self.config.num_episodes_eval):
            episode_reward = 0
            done = False
            state = self._extract_state(self.env.reset())
            while not done:
                state_norm = torch.FloatTensor(state /
                                               self.config.high)
                with torch.no_grad():
                    q_vals = self.q_network(state_norm)
                    action = np.argmax(q_vals.numpy())
                next_state, reward, done, info = self._extract_step(
                    self.env.step(action))
                next_state = None if done else next_state
                state = next_state
                episode_reward += reward

            rewards.append(episode_reward)
        avg_reward = np.mean(rewards)
        std_error = np.sqrt(np.var(rewards) / len(rewards))
        print(f'Eval average reward: {avg_reward:04.2f} +/-'
              f' {std_error:04.2f}')
        sys.stdout.flush()
        return avg_reward
