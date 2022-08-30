import shutil, os, subprocess, cv2
import vista
import numpy as np
import util
import tensorflow as tf
import functools
import tensorflow_probability as tfp

from vista.utils import logging
logging.setLevel(logging.ERROR)

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

memory = Memory()

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

def train_step(model, loss_function, optimizer, observations, actions,
               discounted_rewards, custom_fwd_fn=None):
    with tf.GradientTape() as tape:
        # Forward propagate through the agent network
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

learning_rate = 1e-1
optimizer = tf.keras.optimizers.Adam(learning_rate)
trace_root = "./vista_traces"
trace_path = [
    "20210726-154641_lexus_devens_center", 
    "20210726-155941_lexus_devens_center_reverse", 
    "20210726-184624_lexus_devens_center", 
    "20210726-184956_lexus_devens_center_reverse", 
]
trace_path = [os.path.join(trace_root, p) for p in trace_path]

# Create a virtual world with VISTA, the world is defined by a series of data traces
world = vista.World(trace_path, trace_config={'road_width': 4})

# Create a car in our virtual world. The car will be able to step and take
# different control actions. As the car moves, its sensors will simulate any
# changes it environment
car = world.spawn_agent(
    config={
        'length': 5.,
        'width': 2.,
        'wheel_base': 2.78,
        'steering_ratio': 14.7,
        'lookahead_road': True
    })


smoothed_reward = util.LossHistory(smoothing_factor=0.95)
# Create a camera on the car for synthesizing the sensor data that we can use to train with! 
camera = car.spawn_camera(config={'size': (200, 320)})

# Define a rendering display so we can visualize the simulated car camera
# stream and also get see its physical location with respect to the road in
# its environment. 
display = vista.Display(world, display_config={"gui_scale": 2, "vis_full_frame": False})

# Define a simple helper function that allows us to reset VISTA and the rendering display
def vista_reset():
    world.reset()
    display.reset()

vista_reset()

def vista_step(curvature=None, speed=None):
    # Arguments:
    #   curvature: curvature to step with
    #   speed: speed to step with
    if curvature is None: 
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None: 
        speed = car.trace.f_speed(car.timestamp)
    
    car.step_dynamics(action=np.array([curvature, speed]), dt=1/15.)
    car.step_sensors()

# vista_reset()
# stream = VideoStream()

# for i in range(100):
#     vista_step()
#     vis_img = display.render()
#     stream.write(vis_img[:, :, ::-1], index=i)
#     if car.done: 
#         break

# print("Saving trajectory of human following...")
# stream.save("human_follow.mp4")      

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

# i = 0
# num_crashes = 0
# stream = VideoStream()

# for _ in range(num_crashes):
#     vista_reset()

#     while not check_crash(car):
#         # Sample a random curvature (between +/- 1/3), keep speed constant
#         curvature = np.random.uniform(-1/3, 1/3)
#         # Step the simulated car with the same action
#         vista_step(curvature=curvature)
#         # Render and save the display
#         vis_img = display.render()
#         stream.write(vis_img[:, :, ::-1], index=i)
#         i += 1
    
#     print(f"Car crashed on step {i}")
#     for _ in range(5):
#         stream.write(vis_img[:, :, ::-1], index=i)
#         i += 1

# print("Saving trajectory with random policy...")
# stream.save("random_policy.mp4")

def preprocess(full_obs):
    i1, j1, i2, j2 = camera.camera_param.get_roi()
    obs = full_obs[i1:i2, j1:j2]
    obs = obs / 255.
    return obs

def grab_and_preprocess_obs(car):
    full_obs = car.observations[camera.name]
    obs = preprocess(full_obs)
    return obs

act = tf.keras.activations.swish
Conv2D = functools.partial(tf.keras.layers.Conv2D,
                           padding='valid',
                           activation=act)
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense

def create_driving_model():
    model = tf.keras.models.Sequential([
        Conv2D(filters=32, kernel_size=5, strides=2),
        Conv2D(filters=48, kernel_size=5, strides=2),
        Conv2D(filters=64, kernel_size=3, strides=2),
        Flatten(),
        Dense(units=128, activation=act),
        Dense(units=2, activation=None)
    ])
    return model

driving_model = create_driving_model()
max_curvature = 1/8. 
max_std = 0.1 

def run_driving_model(image):
    # Arguments:
    #   image: an input image
    # Returns:
    #   pred_dist: predicted distribution of control actions 
    single_image_input = tf.rank(image) == 3  # missing 4th batch dimension
    if single_image_input:
        image = tf.expand_dims(image, axis=0)
    distribution = driving_model(image)
    mu, logsigma = tf.split(distribution, 2, axis=1)
    mu = max_curvature * tf.tanh(mu) # conversion
    sigma = max_std * tf.sigmoid(logsigma) + 0.005 # conversion
    pred_dist = tfp.distributions.Normal(mu, sigma)
    return pred_dist

def compute_driving_loss(dist, actions, rewards):
    # Arguments:
    #   logits: network's predictions for actions to take
    #   actions: the actions the agent took in an episode
    #   rewards: the rewards the agent received in an episode
    # Returns:
    #   loss
    neg_logprob = -dist.log_prob(actions)
    loss = tf.reduce_mean(neg_logprob*rewards)
    return loss

max_batch_size = 300
max_reward = float('-inf')

for i_episode in range(500):
    print(i_episode)
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
            train_step(driving_model, compute_driving_loss, optimizer, 
                       observations=np.array(memory.observations)[i],
                       actions=np.array(memory.actions)[i],
                       discounted_rewards = discount_rewards(memory.rewards)[i],
                       custom_fwd_fn=run_driving_model)            
            # reset the memory
            memory.clear()
            break

i_step = 0
num_episodes = 2
num_reset = 2
stream = VideoStream()
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
        i_step += 1
        episode_step += 1
        
    for _ in range(num_reset):
        stream.write(np.zeros_like(vis_img), index=i_step)
        i_step += 1
        
stream.save("trained_policy.mp4")
