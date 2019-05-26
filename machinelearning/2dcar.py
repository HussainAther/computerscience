import numpy as np
import tensorflow as tf
import pyglet
import os
import shutil

"""
2D (two-dimensional 2-dimensional 2 dimensional) car driving.
"""

pyglet.clock.set_fps_limit(10000)

class CarEnv(object):
    n_sensor = 5
    action_dim = 1
    state_dim = n_sensor
    viewer = None
    viewer_xy = (500, 500)
    sensor_max = 150.
    start_point = [450, 300]
    speed = 50.
    dt = 0.1

    def __init__(self, discrete_action=False):
        self.is_discrete_action = discrete_action
        if discrete_action:
            self.actions = [-1, 0, 1]
        else:
            self.action_bound = [-1, 1]

        self.terminal = False
        # node1 (x, y, r, w, l),
        self.car_info = np.array([0, 0, 0, 20, 40], dtype=np.float64)   # car coordination
        self.obstacle_coords = np.array([
            [120, 120],
            [380, 120],
            [380, 380],
            [120, 380],
        ])
        self.sensor_info = self.sensor_max + np.zeros((self.n_sensor, 3))  # n sensors, (distance, end_x, end_y)

    def step(self, action):
        if self.is_discrete_action:
            action = self.actions[action]
        else:
            action = np.clip(action, *self.action_bound)[0]
        self.car_info[2] += action * np.pi/30  # max r = 6 degree
        self.car_info[:2] = self.car_info[:2] + \
                            self.speed * self.dt * np.array([np.cos(self.car_info[2]), np.sin(self.car_info[2])])

        self._update_sensor()
        s = self._get_state()
        r = -1 if self.terminal else 0
        return s, r, self.terminal

    def reset(self):
        self.terminal = False
        self.car_info[:3] = np.array([*self.start_point, -np.pi/2])
        self._update_sensor()
        return self._get_state()

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.car_info, self.sensor_info, self.obstacle_coords)
        self.viewer.render()

    def sample_action(self):
        if self.is_discrete_action:
            a = np.random.choice(list(range(3)))
        else:
            a = np.random.uniform(*self.action_bound, size=self.action_dim)
        return a

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    def _get_state(self):
        s = self.sensor_info[:, 0].flatten()/self.sensor_max
        return s

    def _update_sensor(self):
        cx, cy, rotation = self.car_info[:3]

        n_sensors = len(self.sensor_info)
        sensor_theta = np.linspace(-np.pi / 2, np.pi / 2, n_sensors)
        xs = cx + (np.zeros((n_sensors, ))+self.sensor_max) * np.cos(sensor_theta)
        ys = cy + (np.zeros((n_sensors, ))+self.sensor_max) * np.sin(sensor_theta)
        xys = np.array([[x, y] for x, y in zip(xs, ys)])    # shape (5 sensors, 2)

        # sensors
        tmp_x = xys[:, 0] - cx
        tmp_y = xys[:, 1] - cy
        # apply rotation
        rotated_x = tmp_x * np.cos(rotation) - tmp_y * np.sin(rotation)
        rotated_y = tmp_x * np.sin(rotation) + tmp_y * np.cos(rotation)
        # rotated x y
        self.sensor_info[:, -2:] = np.vstack([rotated_x+cx, rotated_y+cy]).T

        q = np.array([cx, cy])
        for si in range(len(self.sensor_info)):
            s = self.sensor_info[si, -2:] - q
            possible_sensor_distance = [self.sensor_max]
            possible_intersections = [self.sensor_info[si, -2:]]

            # obstacle collision
            for oi in range(len(self.obstacle_coords)):
                p = self.obstacle_coords[oi]
                r = self.obstacle_coords[(oi + 1) % len(self.obstacle_coords)] - self.obstacle_coords[oi]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))

            # window collision
            win_coord = np.array([
                [0, 0],
                [self.viewer_xy[0], 0],
                [*self.viewer_xy],
                [0, self.viewer_xy[1]],
                [0, 0],
            ])
            for oi in range(4):
                p = win_coord[oi]
                r = win_coord[(oi + 1) % len(win_coord)] - win_coord[oi]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = p + t * r
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(intersection - q))

            distance = np.min(possible_sensor_distance)
            distance_index = np.argmin(possible_sensor_distance)
            self.sensor_info[si, 0] = distance
            self.sensor_info[si, -2:] = possible_intersections[distance_index]
            if distance < self.car_info[-1]/2:
                self.terminal = True

class Viewer(pyglet.window.Window):
    color = {
        "background": [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    def __init__(self, width, height, car_info, sensor_info, obstacle_coords):
        super(Viewer, self).__init__(width, height, resizable=False, caption="2D car", vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color["background"])

        self.car_info = car_info
        self.sensor_info = sensor_info

        self.batch = pyglet.graphics.Batch()
        background = pyglet.graphics.OrderedGroup(0)
        foreground = pyglet.graphics.OrderedGroup(1)

        self.sensors = []
        line_coord = [0, 0] * 2
        c = (73, 73, 73) * 2
        for i in range(len(self.sensor_info)):
            self.sensors.append(self.batch.add(2, pyglet.gl.GL_LINES, foreground, ("v2f", line_coord), ("c3B", c)))

        car_box = [0, 0] * 4
        c = (249, 86, 86) * 4
        self.car = self.batch.add(4, pyglet.gl.GL_QUADS, foreground, ("v2f", car_box), ("c3B", c))

        c = (134, 181, 244) * 4
        self.obstacle = self.batch.add(4, pyglet.gl.GL_QUADS, background, ("v2f", obstacle_coords.flatten()), ("c3B", c))

    def render(self):
        pyglet.clock.tick()
        self._update()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event("on_draw")
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        # self.fps_display.draw()

    def _update(self):
        cx, cy, r, w, l = self.car_info

        # sensors
        for i, sensor in enumerate(self.sensors):
            sensor.vertices = [cx, cy, *self.sensor_info[i, -2:]]

        # car
        xys = [
            [cx + l / 2, cy + w / 2],
            [cx - l / 2, cy + w / 2],
            [cx - l / 2, cy - w / 2],
            [cx + l / 2, cy - w / 2],
        ]
        r_xys = []
        for x, y in xys:
            tempX = x - cx
            tempY = y - cy
            # apply rotation
            rotatedX = tempX * np.cos(r) - tempY * np.sin(r)
            rotatedY = tempX * np.sin(r) + tempY * np.cos(r)
            # rotated x y
            x = rotatedX + cx
            y = rotatedY + cy
            r_xys += [x, y]
        self.car.vertices = r_xys

np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 500
MAX_EP_STEPS = 600
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 800
REPLACE_ITER_C = 700
MEMORY_CAPACITY = 2000
BATCH_SIZE = 16
VAR_MIN = 0.1
RENDER = True
LOAD = False
DISCRETE_ACTION = False

env = CarEnv(discrete_action=DISCRETE_ACTION)
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound

# all placeholder for tf
with tf.name_scope("S"):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name="s")
with tf.name_scope("R"):
    R = tf.placeholder(tf.float32, [None, 1], name="r")
with tf.name_scope("S_"):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name="s_")

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
            net = tf.layers.dense(s, 100, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name="l1",
                                  trainable=trainable)
            net = tf.layers.dense(net, 20, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name="l2",
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
                n_l1 = 100
                w1_s = tf.get_variable("w1_s", [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable("w1_a", [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable("b1", [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 20, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name="l2",
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

sess = tf.Session()

# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.train.Saver()
path = "./discrete" if DISCRETE_ACTION else "./continuous"

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())

def train():
    var = 2.  # control exploration
    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_step = 0

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
                var = max([var*.9995, VAR_MIN])    # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_step += 1

            if done or t == MAX_EP_STEPS - 1:
            # if done:
                print("Ep:", ep,
                      "| Steps: %i" % int(ep_step),
                      "| Explore: %.2f" % var,
                      )
                break

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join(path, "DDPG.ckpt")
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)

def eval():
    env.set_fps(30)
    while True:
        s = env.reset()
        while True:
            env.render()
            a = actor.choose_action(s)
            s_, r, done = env.step(a)
            s = s_
            if done:
                break

def intersection():
    p = np.array([0, 0])
    r = np.array([1, 1])
    q = np.array([0.1, 0.1])
    s = np.array([.1, .1])

    if np.cross(r, s) == 0 and np.cross((q-p), r) == 0:    # collinear
        # t0 = (q − p) · r / (r · r)
        # t1 = (q + s − p) · r / (r · r) = t0 + s · r / (r · r)
        t0 = np.dot(q-p, r)/np.dot(r, r)
        t1 = t0 + np.dot(s, r)/np.dot(r, r)
        print(t1, t0)
        if ((np.dot(s, r) > 0) and (0 <= t1 - t0 <= 1)) or ((np.dot(s, r) <= 0) and (0 <= t0 - t1 <= 1)):
            print("collinear and overlapping, q_s in p_r")
        else:
            print("collinear and disjoint")
    elif np.cross(r, s) == 0 and np.cross((q-p), r) != 0:  # parallel r × s = 0 and (q − p) × r ≠ 0,
        print("parallel")
    else:
        t = np.cross((q - p), s) / np.cross(r, s)
        u = np.cross((q - p), r) / np.cross(r, s)
        if 0 <= t <= 1 and 0 <= u <= 1:
            # If r × s ≠ 0 and 0 ≤ t ≤ 1 and 0 ≤ u ≤ 1, the two line segments meet at the point p + t r = q + u s
            print("intersection: ", p + t*r)
        else:
            print("not parallel and not intersect")

def point2segment():
    p = np.array([-1, 1])    # coordination of point
    a = np.array([0, 1])    # coordination of line segment end 1
    b = np.array([1, 0])    # coordination of line segment end 2
    ab = b-a    # line ab
    ap = p-a
    distance = np.abs(np.cross(ab, ap)/np.linalg.norm(ab))  # d = (AB x AC)/|AB|
    print(distance)

    # angle  Cos(θ) = A dot B /(|A||B|)
    bp = p-b
    cosTheta1 = np.dot(ap, ab) / (np.linalg.norm(ap) * np.linalg.norm(ab))
    theta1 = np.arccos(cosTheta1)
    cosTheta2 = np.dot(bp, ab) / (np.linalg.norm(bp) * np.linalg.norm(ab))
    theta2 = np.arccos(cosTheta2)
    if np.pi/2 <= (theta1 % (np.pi*2)) <= 3/2 * np.pi:
        print("out of a")
    elif -np.pi/2 <= (theta2 % (np.pi*2)) <= np.pi/2:
        print("out of b")
    else:
        print("between a and b")

np.random.seed(1)
env = CarEnv()
env.set_fps(30)
for ep in range(20):
    s = env.reset()
    # for t in range(100):
    while True:
        env.render()
        s, r, done = env.step(env.sample_action())
        if done:
            break
train()
eval()
point2segment()
