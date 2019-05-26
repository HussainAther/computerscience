import multiprocessing
import threading
import tensorflow as tf
import numpy as np

"""
Environment is a Robot Arm. The arm tries to get to the blue point.
The environment will return a geographic (distance) information for the arm to learn.
The far away from blue point the less reward; touch blue r+=1; stop at blue for a while then get r=+10.
Uses simple version of OpenAI"s Proximal Policy Optimization (PPO). (http://adsabs.harvard.edu/abs/2017arXiv170706347S)
(DPPO deep ppo), actor-critic, and ddpg. Distributing workers in parallel to collect data, then stop worker"s 
roll-out and train PPO on collected data. Restart workers once PPO is updated.
"""

MAX_GLOBAL_EP = 2000
MAX_EP_STEP = 300
UPDATE_GLOBAL_ITER = 5
N_WORKERS = multiprocessing.cpu_count()
LR_A = 1e-4  # learning rate for actor
LR_C = 2e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
MODE = ["easy", "hard"]
n_model = 1
GLOBAL_NET_SCOPE = "Global_Net"
ENTROPY_BETA = 0.01
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

env = ArmEnv(mode=MODE[n_model])
N_S = env.state_dim
N_A = env.action_dim
A_BOUND = env.action_bound
del env

class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], "S")
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + "/actor")
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + "/critic")
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], "S")
                self.a_his = tf.placeholder(tf.float32, [None, N_A], "A")
                self.v_target = tf.placeholder(tf.float32, [None, 1], "Vtarget")

                mu, sigma, self.v = self._build_net()

                td = tf.subtract(self.v_target, self.v, name="TD_error")
                with tf.name_scope("c_loss"):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope("wrap_a_out"):
                    self.test = sigma[0]
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-5

                normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                with tf.name_scope("a_loss"):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope("choose_a"):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *A_BOUND)
                with tf.name_scope("local_grad"):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + "/actor")
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + "/critic")
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope("sync"):
                with tf.name_scope("pull"):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope("push"):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        w_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("actor"):
            l_a = tf.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init, name="la")
            l_a = tf.layers.dense(l_a, 300, tf.nn.relu6, kernel_initializer=w_init, name="la2")
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name="mu")
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name="sigma")
        with tf.variable_scope("critic"):
            l_c = tf.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init, name="lc")
            l_c = tf.layers.dense(l_c, 200, tf.nn.relu6, kernel_initializer=w_init, name="lc2")
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name="v")  # state value
        return mu, sigma, v

    def update_global(self, feed_dict):  # run by a local
        _, _, t = SESS.run([self.update_a_op, self.update_c_op, self.test], feed_dict)  # local grads applies to global net
        return t

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})[0]

class Worker(object):
    def __init__(self, name, globalAC):
        self.env = ArmEnv(mode=MODE[n_model])
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                if self.name == "W_0":
                    self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done = self.env.step(a)
                if ep_t == MAX_EP_STEP - 1: done = True
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    test = self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                        "| Var:", test,

                          )
                    GLOBAL_EP += 1
                    break

SESS = tf.Session()

with tf.device("/cpu:0"):
    OPT_A = tf.train.RMSPropOptimizer(LR_A, name="RMSPropA")
    OPT_C = tf.train.RMSPropOptimizer(LR_C, name="RMSPropC")
    GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
    workers = []
    # Create worker
    for i in range(N_WORKERS):
        i_name = "W_%i" % i   # worker name
        workers.append(Worker(i_name, GLOBAL_AC))

class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope("Actor"):
            # input s, output a
            self.a = self._build_net(S, scope="eval_net", trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope="target_net", trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/eval_net")
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/target_net")

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name="l1",
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name="l2",
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name="l3",
                                  trainable=trainable)
            with tf.variable_scope("a"):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name="a", trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name="scaled_a")  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope("policy_grads"):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope("A_train"):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope("Critic"):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, "eval_net", trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, "target_net", trainable=False)    # target_q is based on a_ from Actor"s target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Critic/eval_net")
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Critic/target_net")

        with tf.variable_scope("target_q"):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope("TD_error"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope("C_train"):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope("a_grad"):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope("l1"):
                n_l1 = 200
                w1_s = tf.get_variable("w1_s", [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable("w1_a", [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable("b1", [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name="l2",
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name="l3",
                                  trainable=trainable)
            with tf.variable_scope("q"):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, "Memory has not been fulfilled"
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

def train():
    var = 2.  # control exploration

    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0

        for t in range(MAX_EP_STEPS):
        # while True:
            if RENDER:
                env.render()

            # Added exploration noise
            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), *ACTION_BOUND)    # add randomness to action selection for exploration
            s_, r, done = env.step(a)
            M.store_transition(s, a, r, s_)

            if M.pointer > MEMORY_CAPACITY:
                var = max([var*.9999, VAR_MIN])    # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_reward += r

            if t == MAX_EP_STEPS-1 or done:
            # if done:
                result = "| done" if done else "| ----"
                print("Ep:", ep,
                      result,
                      "| R: %i" % int(ep_reward),
                      "| Explore: %.2f" % var,
                      )
                break

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join("./"+MODE[n_model], "DDPG.ckpt")
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)

def eval():
    env.set_fps(30)
    s = env.reset()
    while True:
        if RENDER:
            env.render()
        a = actor.choose_action(s)
        s_, r, done = env.step(a)
        s = s_

class PPO(object):
    def __init__(self):
        self.sess = tf.Session()

        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], "state")

        # critic
        l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
        self.v = tf.layers.dense(l1, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], "discounted_r")
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet("pi", trainable=True)
        oldpi, oldpi_params = self._build_anet("oldpi", trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], "action")
        self.tfadv = tf.placeholder(tf.float32, [None, 1], "advantage")
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv   # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()         # wait until get batch of data
                self.sess.run(self.update_oldpi_op)   # old pi to pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            mu = A_BOUND * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

class ArmEnv(object):
    action_bound = [-1, 1]
    action_dim = 2
    state_dim = 7
    dt = .1  # refresh rate
    arm1l = 100
    arm2l = 100
    viewer = None
    viewer_xy = (400, 400)
    get_point = False
    mouse_in = np.array([False])
    point_l = 15
    grab_counter = 0

    def __init__(self, mode="easy"):
        # node1 (l, d_rad, x, y),
        # node2 (l, d_rad, x, y)
        self.mode = mode
        self.arm_info = np.zeros((2, 4))
        self.arm_info[0, 0] = self.arm1l
        self.arm_info[1, 0] = self.arm2l
        self.point_info = np.array([250, 303])
        self.point_info_init = self.point_info.copy()
        self.center_coord = np.array(self.viewer_xy)/2

    def step(self, action):
        # action = (node1 angular v, node2 angular v)
        action = np.clip(action, *self.action_bound)
        self.arm_info[:, 1] += action * self.dt
        self.arm_info[:, 1] %= np.pi * 2

        arm1rad = self.arm_info[0, 1]
        arm2rad = self.arm_info[1, 1]
        arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)])
        arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2rad), self.arm_info[1, 0] * np.sin(arm2rad)])
        self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy  # (x1, y1)
        self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2dx_dy  # (x2, y2)

        s, arm2_distance = self._get_state()
        r = self._r_func(arm2_distance)

        return s, r, self.get_point

    def reset(self):
        self.get_point = False
        self.grab_counter = 0

        if self.mode == "hard":
            pxy = np.clip(np.random.rand(2) * self.viewer_xy[0], 100, 300)
            self.point_info[:] = pxy
        else:
            arm1rad, arm2rad = np.random.rand(2) * np.pi * 2
            self.arm_info[0, 1] = arm1rad
            self.arm_info[1, 1] = arm2rad
            arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)])
            arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2rad), self.arm_info[1, 0] * np.sin(arm2rad)])
            self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy  # (x1, y1)
            self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2dx_dy  # (x2, y2)

            self.point_info[:] = self.point_info_init
        return self._get_state()[0]

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.arm_info, self.point_info, self.point_l, self.mouse_in)
        self.viewer.render()

    def sample_action(self):
        return np.random.uniform(*self.action_bound, size=self.action_dim)

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    def _get_state(self):
        # return the distance (dx, dy) between arm finger point with blue point
        arm_end = self.arm_info[:, 2:4]
        t_arms = np.ravel(arm_end - self.point_info)
        center_dis = (self.center_coord - self.point_info)/200
        in_point = 1 if self.grab_counter > 0 else 0
        return np.hstack([in_point, t_arms/200, center_dis,
                          # arm1_distance_p, arm1_distance_b,
                          ]), t_arms[-2:]

    def _r_func(self, distance):
        t = 50
        abs_distance = np.sqrt(np.sum(np.square(distance)))
        r = -abs_distance/200
        if abs_distance < self.point_l and (not self.get_point):
            r += 1.
            self.grab_counter += 1
            if self.grab_counter > t:
                r += 10.
                self.get_point = True
        elif abs_distance > self.point_l:
            self.grab_counter = 0
            self.get_point = False
        return r

class Viewer(pyglet.window.Window):
    color = {
        "background": [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    def __init__(self, width, height, arm_info, point_info, point_l, mouse_in):
        super(Viewer, self).__init__(width, height, resizable=False, caption="Arm", vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color["background"])

        self.arm_info = arm_info
        self.point_info = point_info
        self.mouse_in = mouse_in
        self.point_l = point_l

        self.center_coord = np.array((min(width, height)/2, ) * 2)
        self.batch = pyglet.graphics.Batch()

        arm1_box, arm2_box, point_box = [0]*8, [0]*8, [0]*8
        c1, c2, c3 = (249, 86, 86)*4, (86, 109, 249)*4, (249, 39, 65)*4
        self.point = self.batch.add(4, pyglet.gl.GL_QUADS, None, ("v2f", point_box), ("c3B", c2))
        self.arm1 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ("v2f", arm1_box), ("c3B", c1))
        self.arm2 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ("v2f", arm2_box), ("c3B", c1))

    def render(self):
        pyglet.clock.tick()
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event("on_draw")
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        # self.fps_display.draw()

    def _update_arm(self):
        point_l = self.point_l
        point_box = (self.point_info[0] - point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] + point_l,
                     self.point_info[0] - point_l, self.point_info[1] + point_l)
        self.point.vertices = point_box

        arm1_coord = (*self.center_coord, *(self.arm_info[0, 2:4]))  # (x0, y0, x1, y1)
        arm2_coord = (*(self.arm_info[0, 2:4]), *(self.arm_info[1, 2:4]))  # (x1, y1, x2, y2)
        arm1_thick_rad = np.pi / 2 - self.arm_info[0, 1]
        x01, y01 = arm1_coord[0] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] + np.sin(
            arm1_thick_rad) * self.bar_thc
        x02, y02 = arm1_coord[0] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] - np.sin(
            arm1_thick_rad) * self.bar_thc
        x11, y11 = arm1_coord[2] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] - np.sin(
            arm1_thick_rad) * self.bar_thc
        x12, y12 = arm1_coord[2] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] + np.sin(
            arm1_thick_rad) * self.bar_thc
        arm1_box = (x01, y01, x02, y02, x11, y11, x12, y12)
        arm2_thick_rad = np.pi / 2 - self.arm_info[1, 1]
        x11_, y11_ = arm2_coord[0] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] - np.sin(
            arm2_thick_rad) * self.bar_thc
        x12_, y12_ = arm2_coord[0] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] + np.sin(
            arm2_thick_rad) * self.bar_thc
        x21, y21 = arm2_coord[2] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] + np.sin(
            arm2_thick_rad) * self.bar_thc
        x22, y22 = arm2_coord[2] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] - np.sin(
            arm2_thick_rad) * self.bar_thc
        arm2_box = (x11_, y11_, x12_, y12_, x21, y21, x22, y22)
        self.arm1.vertices = arm1_box
        self.arm2.vertices = arm2_box

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.UP:
            self.arm_info[0, 1] += .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.DOWN:
            self.arm_info[0, 1] -= .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.LEFT:
            self.arm_info[1, 1] += .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.RIGHT:
            self.arm_info[1, 1] -= .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.Q:
            pyglet.clock.set_fps_limit(1000)
        elif symbol == pyglet.window.key.A:
            pyglet.clock.set_fps_limit(30)

    def on_mouse_motion(self, x, y, dx, dy):
        self.point_info[:] = [x, y]

    def on_mouse_enter(self, x, y):
        self.mouse_in[0] = True

    def on_mouse_leave(self, x, y):
        self.mouse_in[0] = False

COORD = tf.train.Coordinator()
SESS.run(tf.global_variables_initializer())

sess = tf.Session()

# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.train.Saver()
path = "./"+MODE[n_model]

worker_threads = []
for worker in workers:
    job = lambda: worker.work()
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)
COORD.join(worker_threads)

train()
eval()

GLOBAL_PPO = PPO()
UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
UPDATE_EVENT.clear()    # no update now
ROLLING_EVENT.set()     # start to roll out
workers = [Worker(wid=i) for i in range(N_WORKER)]

GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
GLOBAL_RUNNING_R = []
COORD = tf.train.Coordinator()
QUEUE = queue.Queue()
threads = []
for worker in workers:  # worker threads
    t = threading.Thread(target=worker.work, args=())
    t.start()
    threads.append(t)
# add a PPO updating thread
threads.append(threading.Thread(target=GLOBAL_PPO.update,))
threads[-1].start()
COORD.join(threads)

# plot reward change and testing
plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
plt.xlabel("Episode"); plt.ylabel("Moving reward"); plt.ion(); plt.show()
env.set_fps(30)
while True:
    s = env.reset()
    for t in range(400):
        env.render()
        s = env.step(GLOBAL_PPO.choose_action(s))[0]
