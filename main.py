import cv2
import functools
import numpy as np
import os
import shutil
import subprocess
import tensorflow as tf
import tensorflow_probability as tfp
import util
import vista


class Memory:
    def __init__(self): 
        self.clear()

    def clear(self): 
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_to_memory(self, new_observation, new_action, new_reward): 
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

    def __len__(self):
        return len(self.actions)


class VideoStream():
    def __init__(self):
        self.tmp = "./tmp"
        if os.path.exists(self.tmp) and os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp)
        os.mkdir(self.tmp)

    def write(self, image, index):
        cv2.imwrite(os.path.join(self.tmp, f"{index:04}.png"), image)

    def save(self, fname):
        cmd = f"ffmpeg -f image2 -i {self.tmp}/%04d.png -crf 0 -y {fname}"
        subprocess.call(cmd, shell=True)


def vista_reset():
    world.reset()
    display.reset()

def train_step(model, loss_function, optimizer, observations, actions,
               discounted_rewards, custom_fwd_fn=None):
    with tf.GradientTape() as tape:
        if custom_fwd_fn is not None:
            prediction = custom_fwd_fn(observations)
        else: 
            prediction = model(observations)
        loss = loss_function(prediction, actions, discounted_rewards)

    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 2)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)

def discount_rewards(rewards, gamma=0.95): 
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        R = R*gamma + rewards[t]
        discounted_rewards[t] = R
    return normalize(discounted_rewards)

def vista_step(curvature=None, speed=None):
    if curvature is None: 
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None: 
        speed = car.trace.f_speed(car.timestamp)

    car.step_dynamics(action=np.array([curvature, speed]), dt=1/15.)
    car.step_sensors()

def check_out_of_lane(car):
    distance_from_center = np.abs(car.relative_state.x)
    road_width = car.trace.road_width 
    half_road_width = road_width / 2
    return distance_from_center > half_road_width

def check_exceed_max_rot(car):
    maximal_rotation = np.pi / 10.
    current_rotation = np.abs(car.relative_state.yaw)
    return current_rotation > maximal_rotation

def check_crash(car): 
    return check_out_of_lane(car) or check_exceed_max_rot(car) or car.done

def preprocess(full_obs):
    i1, j1, i2, j2 = camera.camera_param.get_roi()
    obs = full_obs[i1:i2, j1:j2]
    obs = obs / 255.
    return obs

def grab_and_preprocess_obs(car):
    full_obs = car.observations[camera.name]
    obs = preprocess(full_obs)
    return obs

def create_driving_model(acts, Conv2D):
    return tf.keras.models.Sequential([
        Conv2D(filters=32, kernel_size=5, strides=2),
        Conv2D(filters=48, kernel_size=5, strides=2),
        Conv2D(filters=64, kernel_size=3, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation=acts),
        tf.keras.layers.Dense(units=2, activation=None)
    ])

def run_driving_model(image):
    single_image_input = tf.rank(image) == 3
    if single_image_input:
        image = tf.expand_dims(image, axis=0)
    distribution = driving_model(image)
    mu, logsigma = tf.split(distribution, 2, axis=1)
    mu = max_curvature*tf.tanh(mu)
    sigma = max_std*tf.sigmoid(logsigma) + 0.005
    pred_dist = tfp.distributions.Normal(mu, sigma)
    return pred_dist

def compute_driving_loss(dist, actions, rewards):
    neg_logprob = -dist.log_prob(actions)
    loss = tf.reduce_mean(neg_logprob*rewards)
    return loss

def train_model(num_episodes=500, max_batch_size=300,
                max_reward=float('-inf')):
    for i_episode in range(num_episodes):
        vista_reset()
        memory.clear()
        observation = grab_and_preprocess_obs(car)

        while True:
            curvature_dist = run_driving_model(observation)
            curvature_action = curvature_dist.sample()[0, 0]
            vista_step(curvature_action)
            observation = grab_and_preprocess_obs(car)
            reward = 10.0 if not check_crash(car) else 0.0
            memory.add_to_memory(observation, curvature_action, reward)
            # is the episode over? did you crash or do so well that you're done?
            if reward == 0.0:
                total_reward = sum(memory.rewards)
                smoothed_reward.append(total_reward)
                batch_size = min(len(memory), max_batch_size)
                i = np.random.choice(len(memory), batch_size, replace=False)
                discounted_rewards = discount_rewards(memory.rewards)[i]
                train_step(driving_model,
                           compute_driving_loss,
                           optimizer, 
                           observations=np.array(memory.observations)[i],
                           actions=np.array(memory.actions)[i],
                           discounted_rewards=discounted_rewards,
                           custom_fwd_fn=run_driving_model)
                memory.clear()
                break

def run_model(stream, i_step=0, num_episodes=1):
    for i_episode in range(num_episodes):
        vista_reset()
        observation = grab_and_preprocess_obs(car)
        episode_step = 0
        while not check_crash(car) and episode_step < 100:
        # using our observation, choose an action and take it in the environment
            curvature_dist = run_driving_model(observation)
            curvature = curvature_dist.mean()[0, 0]

            # Step the simulated car with the same action
            vista_step(curvature)
            observation = grab_and_preprocess_obs(car)
            vis_img = display.render()
            stream.write(vis_img[:, :, ::-1], index=i_step)
            i_step += 1; episode_step += 1


# setting variables
learning_rate = 1
optimizer = tf.keras.optimizers.Adam(learning_rate)
smoothed_reward = util.LossHistory(smoothing_factor=0.95)
trace_root = "./vista_traces"
trace_path = [
    "20210726-154641_lexus_devens_center",
    # "20210726-155941_lexus_devens_center_reverse",
    # "20210726-184624_lexus_devens_center",
    # "20210726-184956_lexus_devens_center_reverse",
]
trace_path = [os.path.join(trace_root, p) for p in trace_path]
world = vista.World(trace_path, trace_config={'road_width': 4})
car = world.spawn_agent(
    config={
        'length': 5.,
        'width': 2.,
        'wheel_base': 2.78,
        'steering_ratio': 14.7,
        'lookahead_road': True
    })
camera = car.spawn_camera(config={'size': (200, 320)})
display = vista.Display(world, display_config={"gui_scale": 2,
                                               "vis_full_frame": False})
acts = tf.keras.activations.swish
Conv2D = functools.partial(tf.keras.layers.Conv2D,
                           padding='valid',
                           activation=acts)
driving_model = create_driving_model(acts, Conv2D)
max_curvature = 0.125
max_std = 0.1


# code execution per se
memory = Memory()
stream = VideoStream()
vista_reset()
exit()
train_model(num_episodes=2)
run_model(stream=stream)
stream.save("trained_policy.mp4")
