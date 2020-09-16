from collections import deque
import random
import tensorflow as tf
import numpy as np

class DDPG():
    """
    This class contains the functions and parameters to implement the DDPG algorithm. 

    """
    def __init__(self, env):

      self.env     = env
      self.memory  = deque(maxlen=100000)
        
      self.gamma = 0.9
      self.epsilon = 1.0
      self.epsilon_min = 0.01 # 0.01
      self.epsilon_decay = 0.98
      self.learning_rate_actor = 0.0001
      self.learning_rate_critic = 0.001

      self.loss_h = []

      self.model_c = self.create_model_critic()
      self.target_model_c = self.create_model_critic()

      self.model_a = self.create_model_actor()
      self.target_model_a = self.create_model_actor()
    
    def create_model_critic(self):
      """
      The function for creating the Q critic network. 

      """

      w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
      input_1 = tf.keras.Input(shape=(self.env.obs_space_n, ))
      input_2 = tf.keras.Input(shape=(self.env.action_space_n, ))
      con = tf.keras.layers.concatenate([input_1, input_2])
      dense = tf.keras.layers.Dense(24, activation='relu')(con)
      dense = tf.keras.layers.Dense(48, activation='relu')(dense)
      output = tf.keras.layers.Dense(1, activation='relu',  kernel_initializer = w_init)(dense)

      model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
      
      model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate_critic))
      return model
    
    def create_model_actor(self):
      """
      The function for creating the mu actor network. 
      
      """

      w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
      input_1 = tf.keras.Input(shape=(self.env.obs_space_n, ))
      dense = tf.keras.layers.Dense(24, activation='relu')(input_1)
      dense = tf.keras.layers.BatchNormalization()(dense)
      dense = tf.keras.layers.Dense(48, activation='relu')(dense)
      dense = tf.keras.layers.BatchNormalization()(dense)
      # dense = tf.keras.layers.Dense(24, activation='relu', kernel_initializer = w_init)(dense)
      # dense = tf.keras.layers.BatchNormalization()(dense)
      out = tf.keras.layers.Dense(self.env.action_space_n, activation='tanh', kernel_initializer = w_init)(dense)
      scaled_out = tf.multiply(out, self.env.action_space_bounds)

      # scaled_out = tf.keras.layers.Dense(self.env.action_space_n, activation= 'linear', kernel_initializer = w_init)(dense)
      model = tf.keras.Model(inputs=input_1, outputs=scaled_out)
      
      self.opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_actor)

      return model

    def remember(self, state, action, reward, new_state, done):
      self.memory.append([state, action, reward, new_state, done])
    
    def grad_critic_a(self, state, action):
      """
      This function calculates the gradient of Q funcion with respect to action.

      """
      x1t=tf.Variable(state) #convert_to_tensor(state)
      x2t=tf.Variable(action) #convert_to_tensor(action)
      with tf.GradientTape() as g:
          g.watch(x2t)
          y = self.model_c([x1t,x2t])
      return g.gradient(y, x2t)
    
    def grad_actor(self, state, grad):
      """
      This function calculates the policy gradient for the actor.

      """

      grad = tf.cast(grad, dtype= tf.float32)
      x1t= tf.Variable(state) #convert_to_tensor(state)
      with tf.GradientTape() as g:
          g.watch(self.model_a.trainable_variables)
          y = self.model_a(x1t)
      return g.gradient(y, self.model_a.trainable_variables, -grad)
      

    def replay(self):
      batch_size = 32
      if len(self.memory) < batch_size: 
          return
      
      samples = random.sample(self.memory, batch_size)
      # samples = random.sample(self.memory, batch_size // 2) + [self.memory[i] for i in range(-batch_size // 2, 0)]
      obs_state = np.empty((batch_size,) + (self.env.obs_space_n,))
      obs_action = np.empty((batch_size,) + (self.env.action_space_n,))
      obs_target = np.empty((batch_size,) + (1,))
      max_grad = 0
      max_grad2 = 0

      for i in range(len(samples)):
          state, action, reward, new_state, done = samples[i]
          # target = self.target_model_c.predict(state)
          if done:
              target = reward
              # print(reward)
          else:
              new_ac = self.target_model_a.predict(new_state)               
              Q_future = self.target_model_c.predict([new_state, new_ac])
              target = reward + Q_future * self.gamma
          obs_state[i] = state
          obs_action[i] = action
          target= np.array(target).reshape(1,1)
          obs_target[i] = target
          # print(state,action,new_state)
          # history = self.model_c.fit([state, action] , target, epochs=1, verbose=0)
          # self.loss_h.append(history.history['loss'])

          grad = self.grad_critic_a(state, action)
          # if np.linalg.norm(grad) > 0.01: print(grad)
          max_grad = max(max_grad, np.linalg.norm(grad))
          if i == 0:
            grad2 = self.grad_actor( state, grad)
          else: 
            grad2 += self.grad_actor( state, grad)
          # grad2 = self.grad_actor( state, grad)
          max_grad2 = max(max_grad2, np.linalg.norm(np.array(grad2)[5]))
          # self.opt.apply_gradients(zip(grad2, self.model_a.trainable_variables))

      self.model_c.train_on_batch([obs_state, obs_action], obs_target)
      grad2 = list(map( lambda x: x/ batch_size , grad2))
      self.opt.apply_gradients(zip(grad2, self.model_a.trainable_variables))
      # if max_grad < 100:
      #     print(max_grad, np.linalg.norm(np.array(grad2)[3]))
      # if max_grad2 < .01:
      #     print(max_grad2)
        

    def target_train_a(self, beta, w_m):
      weights = self.model_a.get_weights()
      target_weights = self.target_model_a.get_weights()
      for i in range(len(target_weights)):
          target_weights[i] = (beta/2) * weights[i] + ((1-beta)/2) * target_weights[i] + 0.5 * w_m[i]
      self.target_model_a.set_weights(target_weights)
    
    def target_train_c(self, beta):
      # beta = 0.5
      weights = self.model_c.get_weights()
      target_weights = self.target_model_c.get_weights()
      for i in range(len(target_weights)):
          target_weights[i] = beta * weights[i] + (1-beta) * target_weights[i]
      self.target_model_c.set_weights(target_weights)

    def act(self, state):
      if np.random.random() < self.epsilon:
        action = self.env.action()
        return action
      action = self.target_model_a.predict(state)[0]     
      return action.reshape(1,self.env.action_space_n)

    def set_eps(self):
      self.epsilon *= self.epsilon_decay
      self.epsilon = max(self.epsilon_min, self.epsilon)


def trajectory(model, initial_state, T):
  """
  This function resutns the sum of rewards for a trajectory of length T starting from initial_state
  
  """
  cur_state = initial_state
  cur_val, flag = model.env.env_LP(cur_state)
  # print(cur_state, cur_val)
  res = 0
  for _ in range(T):
    action = model.model_a.predict(cur_state.reshape(1,10))[0]
    next_state = cur_state + action
    val, flag = model.env.env_LP(next_state)
    # print(next_state, val)
    if flag: break
    res += cur_val - val
    cur_state = next_state
    cur_val = val 
  return res