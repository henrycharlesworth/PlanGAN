import torch
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import os
import sys
import argparse
import json
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist

try:
    from fetch_proof_of_principle.env.fetch_reach_mod import FetchReachMod
    from fetch_proof_of_principle.networks.networks import Generator, Discriminator, Generator_alt, Discriminator_alt
except:
    from env.fetch_reach_mod import FetchReachMod
    from networks.networks import Generator, Discriminator, Generator_alt, Discriminator_alt

parser = argparse.ArgumentParser()
parser.add_argument('--nobatchnorm', dest='batch_norm', action='store_false', help='batchnorm in generator or not')
parser.add_argument('--noSN', dest='spectral_norm', action='store_false', help='apply spectral normalisation')
parser.add_argument('--noGP', dest='grad_penalty', action='store_false', help='apply gradient penalty')
parser.add_argument('--DS', dest='diversity_sensitivity', action='store_true', help='use additional DS loss with generator')
parser.add_argument('--num_discrim_updates', type=int, default=5)
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--b1', type=float, default=0.0)
parser.add_argument('--b2', type=float, default=0.9)
parser.add_argument('--grad_param', type=float, default=10.0)
parser.add_argument('--ds_loss_coeff', type=float, default=1.0)
parser.add_argument('--data_size', type=int, default=1000000)
parser.add_argument('--traj_length', type=int, default=100, help="length of trajectories")
parser.add_argument('--train_steps', type=int, default=1000000)
parser.add_argument('--num_trajs', type=int, default=100)
parser.add_argument('--eval_every', type=int, default=10000)
parser.add_argument('--tau', type=int, default=5, help="trying to learn states tau steps into future")
parser.add_argument('--addskipconnections', dest='skip_connections', action='store_true')
parser.add_argument('--experiment_name', type=str, default="default")
parser.add_argument('--discriminator_clamp_val', type=float, default=1.0)
parser.add_argument('--discriminator_clamp', dest='clamp_d', action='store_true')
parser.add_argument('--generator_clamp', dest='clamp_g', action='store_true')
parser.add_argument('--generator_clamp_val', type=float, default=3.0)
parser.add_argument('--generator_tanh', dest='tanh_g', action='store_true')
parser.add_argument('--generator_tanh_value', type=float, default=3.0)
parser.add_argument('--alternative_ds_normalisation', dest='alt_ds_norm', action='store_true')
parser.add_argument('--l2reg_d', dest='l2_reg_d', action='store_true')
parser.add_argument('--l2reg_g', dest='l2_reg_g', action='store_true')
parser.add_argument('--l2regparam', type=float, default=0.0001)

parser.add_argument('--h1', type=int, default=0)
parser.add_argument('--h2', type=int, default=0)
parser.add_argument('--h3', type=int, default=0)
parser.set_defaults(batch_norm=True, spectral_norm=True, grad_penalty=True, diversity_sensitivity=False,
                    skip_connections=False, clamp_d=False, clamp_g=False, tanh_g=False, alt_ds_norm=False,
                    l2_reg_d=False, l2_reg_g=False)

args = parser.parse_args()

num_discrim_updates = args.num_discrim_updates
latent_dim = args.latent_dim
batch_size = args.batch_size
data_size = args.data_size
traj_length = args.traj_length
train_steps = args.train_steps
tau = args.tau
num_trajs = args.num_trajs
experiment_name = args.experiment_name
state_dim = 3 #just use goals as input/output

os.makedirs("results", exist_ok=True)
os.makedirs("results/nearby_states", exist_ok=True)
os.makedirs("results/nearby_states/"+experiment_name, exist_ok=True)
json.dump(vars(args), open("results/nearby_states/"+experiment_name+"/parameters.json", "w"))

device = "cuda" if torch.cuda.is_available() else "cpu"

"""GENERATE/LOAD DATA"""
possible_start_states = joblib.load("starting_states.pkl")
env = FetchReachMod(random_start=True, possible_start_states=possible_start_states)
env.display_goal_marker = False
env.reset()
#env.render() #have to do this to generate images -_-
#env.viewer._hide_overlay = True
#env.viewer.vopt.geomgroup[0] ^= 1 #hide robot/table
if os.path.exists("data"+str(num_trajs)+"_"+str(traj_length)+".pkl"):
    data = joblib.load("data"+str(num_trajs)+"_"+str(traj_length)+".pkl")
else:
    data = np.zeros((num_trajs, traj_length, state_dim))
    for t in range(num_trajs):
        obs = env.reset()
        data[t,0,:] = obs["achieved_goal"]
        for s in range(1, traj_length):
            obs, _, _, _ = env.step(env.action_space.sample())
            data[t,s,:] = obs["achieved_goal"]
        print("Traj %d generated" % t)
    joblib.dump(data, "data"+str(num_trajs)+"_"+str(traj_length)+".pkl")
    print("Data generated successfully!")

num_val = 100
data_val = np.zeros((num_val, traj_length, state_dim))
for t in range(num_val):
    obs = env.reset()
    data_val[t,0,:] = obs["achieved_goal"]
    for s in range(1, traj_length):
        obs, _, _, _ = env.step(env.action_space.sample())
        data_val[t,s,:] = obs["achieved_goal"]
    print("Val traj %d generated" % t)

scaler = StandardScaler()
data = scaler.fit_transform(data.reshape(-1, state_dim)).reshape(num_trajs, traj_length, state_dim)
data_val = scaler.transform(data_val.reshape(-1, state_dim)).reshape(num_val, traj_length, state_dim)

s_ind = np.array(range(0, traj_length-tau))
t_ind = np.array([i for i in range(num_trajs) for _ in range(len(s_ind))])
s_ind = np.tile(s_ind, num_trajs)

states = data[t_ind, s_ind, :]
f_states = data[t_ind, s_ind + tau, :]
distances = np.linalg.norm((states-f_states).reshape(-1, state_dim), axis=-1)
avg_distance_emp_tau = np.mean(distances)
std_distance_emp_tau = np.std(distances)

def sample_batch(size):
    traj_ind = np.random.randint(0, num_trajs, size)
    state_ind = np.random.randint(0, traj_length-tau, size)
    future_state_ind = state_ind + tau
    return torch.tensor(data[traj_ind, state_ind, ...], dtype=torch.float32).to(device), \
           torch.tensor(data[traj_ind, future_state_ind, ...], dtype=torch.float32).to(device)

def sample_batch_val(size):
    traj_ind = np.random.randint(0, num_val, size)
    state_ind = np.random.randint(0, traj_length - tau, size)
    future_state_ind = state_ind + tau
    return torch.tensor(data_val[traj_ind, state_ind, ...], dtype=torch.float32).to(device), \
           torch.tensor(data_val[traj_ind, future_state_ind, ...], dtype=torch.float32).to(device)

"""GENERATOR/DISCRIMINATOR NETWORKS"""
if args.h1 == 0:
    hidden_sizes_G = [128, 256, 512]
    hidden_sizes_D = [512, 256, 128]
else:
    hidden_sizes_G = [args.h1]
    if args.h2 != 0:
        hidden_sizes_G.append(args.h2)
    if args.h3 != 0:
        hidden_sizes_G.append(args.h3)
    hidden_sizes_D = list(reversed(hidden_sizes_G))
if args.clamp_d:
    clamp_d = True
    clamp_d_val = args.discriminator_clamp_val
else:
    clamp_d = False
    clamp_d_val = None
if args.clamp_g:
    clamp_g = True
    clamp_g_val = args.generator_clamp_val
else:
    clamp_g = False
    clamp_g_val = None
if args.tanh_g:
    tanh_g = True
    tanh_g_val = args.generator_tanh_value
else:
    tanh_g = False
    tanh_g_val = None

if args.skip_connections:
    netG = Generator_alt(state_dim, latent_dim, state_dim, batch_norm=args.batch_norm, hidden_sizes=hidden_sizes_G,
                         clamp=clamp_g, clamp_value=clamp_g_val, tanh=tanh_g, tanh_value=tanh_g_val).to(device)
    netD = Discriminator_alt(state_dim, spectral_norm=args.spectral_norm, hidden_sizes=hidden_sizes_D,
                             clamp=clamp_d, clamp_value=clamp_d_val).to(device)
else:
    netG = Generator(state_dim, latent_dim, state_dim, batch_norm=args.batch_norm, hidden_sizes=hidden_sizes_G,
                     clamp=clamp_g, clamp_value=clamp_g_val, tanh=tanh_g, tanh_value=tanh_g_val).to(device)
    netD = Discriminator(state_dim, spectral_norm=args.spectral_norm, hidden_sizes=hidden_sizes_D,
                         clamp=clamp_d, clamp_value=clamp_d_val).to(device)
optimiser_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimiser_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.b1, args.b2))

"""RECORD LOSS"""
loss_record = {"G": [], "D": []}
if args.diversity_sensitivity:
    loss_record["DS"] = []
    loss_record["G_tot"] = []
if args.grad_penalty:
    loss_record["GP"] = []
    loss_record["D_tot"] = []
loss_record["avg_distance_empirical"] = avg_distance_emp_tau
loss_record["std_distance_empirical"] = std_distance_emp_tau
loss_record["avg_distance"] = []
loss_record["std_distance"] = []
loss_record["big_eval_avg_distance"] = []
loss_record["big_eval_std_distance"] = []
loss_record["big_eval_avg_nn_distance"] = []
loss_record["generator_grad_norm"] = []
loss_record["discriminator_grad_norm"] = []

"""FOR GENERATING SAMPLE IMAGES"""
def generate_sample_states(num_images, num_future_states, epoch):
    os.makedirs("results/nearby_states/"+experiment_name+"/images", exist_ok=True)
    current_states, future_states = sample_batch(num_images)
    fig = plt.figure(figsize=(18,3))
    axes = []
    for i in range(num_images):
        axes.append(fig.add_subplot(1,num_images,i+1,projection='3d'))
    fig.suptitle("Examples after " + str(epoch) + " training steps")
    for i in range(num_images):
        """
        env.reset_markers()
        #current state marker
        env.add_markers([{"pos": current_states[i,:].cpu().data.numpy(),
                         "rgba": [0.0, 1.0, 0.0, 1.0],
                         "type": env.marker_type,
                         "size": env.marker_size}], use_default_settings=False)
        #future state marker
        env.add_markers([{"pos": future_states[i,:].cpu().data.numpy(),
                         "rgba": [1.0, 0.0, 0.0, 1.0],
                         "type": env.marker_type,
                         "size": env.marker_size}], use_default_settings=False)
        """
        c_state = scaler.inverse_transform(current_states[i,:].cpu().data.numpy())
        f_state = scaler.inverse_transform(future_states[i,:].cpu().data.numpy())

        axes[i].scatter(c_state[0], c_state[1], c_state[2], c="green", s=45)
        axes[i].scatter(f_state[0], f_state[1], f_state[2], c="red", s=45)

        z = torch.randn(num_future_states, latent_dim).to(device)
        netG.eval()
        gen_future_states = netG(z, current_states[i,:].repeat(num_future_states,1))
        netG.train()
        #gen_future_states = current_states[i,:].repeat(num_future_states,1) + 0.05*torch.randn_like(current_states[i,:].repeat(num_future_states,1)).to(device)
        g_states = scaler.inverse_transform(gen_future_states.cpu().data.numpy())
        axes[i].scatter(g_states[:,0], g_states[:,1], g_states[:,2], c="blue", s=25)
        #gen_markers = [gen_future_states[p].cpu().data.numpy() for p in range(num_future_states)]
        #env.add_markers(gen_markers)
        #env.render()
        #img=env.viewer._read_pixels_as_in_window(resolution=(1024,1024))
        #axes[i].imshow(img)
    filename = "results/nearby_states/"+experiment_name+"/images/generated_samples_epoch_"+str(epoch)+".png"
    plt.savefig(filename)

def big_eval(batch_size, num_future_states):
    states, _ = sample_batch_val(batch_size)
    states = states.unsqueeze(1).repeat(1, num_future_states, 1).view(-1, state_dim)
    z = torch.randn(states.size(0), latent_dim).to(device)
    netG.eval()
    gen_future_states = netG(z, states)
    netG.train()
    distances = (states-gen_future_states).norm(dim=-1)
    loss_record["big_eval_avg_distance"].append(distances.mean().item())
    loss_record["big_eval_std_distance"].append(distances.std().item())
    c_ind = 0
    median_pairwise_distances = []
    for i in range(batch_size):
        data = gen_future_states[c_ind:c_ind+num_future_states, ...].cpu().data.numpy().astype(float)
        pairwise_distances = pdist(data)
        median_pairwise_distances.append(np.median(pairwise_distances))

    mean_median_pairwise_distances = np.mean(np.array(median_pairwise_distances))
    loss_record["big_eval_avg_nn_distance"].append(mean_median_pairwise_distances)

"""MAIN TRAINING"""
for epoch in range(train_steps):

    current_states, future_states = sample_batch(batch_size)

    """TRAIN DISCRIMINATOR"""
    optimiser_D.zero_grad()

    z = torch.randn(batch_size, latent_dim, dtype=torch.float32).to(device)
    gen_future_states = netG(z, current_states)
    errD_real = netD(current_states, future_states)
    errD_fake = netD(current_states, gen_future_states)

    loss_D_p1 = errD_real.mean() - errD_fake.mean()
    #gradient penalty
    if args.grad_penalty:
        alpha = torch.tensor(np.random.random((batch_size, 1)), dtype=torch.float32).to(device)
        interpolates = (alpha*future_states + (1-alpha)*gen_future_states)
        d_interpolates = netD(current_states, interpolates)
        g_out = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=g_out, create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        grad_loss = args.grad_param * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        loss_D = loss_D_p1 + grad_loss
    else:
        loss_D = loss_D_p1
    if args.l2_reg_d:
        p_d = 0
        for p in netD.parameters():
            p_d += torch.dot(p.view(-1), p.view(-1))
        loss_D += p_d*args.l2regparam
    loss_D.backward()
    total_norm = 0
    for p in netD.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item()**2
    loss_record["discriminator_grad_norm"].append(total_norm**(1.0/2))
    loss_record["D"].append(loss_D_p1.item())
    str_to_print = "[Epoch: %d/%d] [D_loss (base): %f] " % (epoch, train_steps, loss_D_p1.item())
    if args.grad_penalty:
        loss_record["GP"].append(grad_loss.item())
        loss_record["D_tot"].append(loss_D.item())
        str_to_print += "[D_loss (tot): %f]" % (loss_D.item())
    optimiser_D.step()

    distances = (current_states - gen_future_states).norm(dim=-1)
    avg_distance = torch.mean(distances).item()
    std_distance = torch.std(distances).item()
    loss_record["avg_distance"].append(avg_distance)
    loss_record["std_distance"].append(std_distance)
    str_to_print += "[avg distance: %f (empirical: %f)] [std distance: %f (empirical: %f)] " % \
                    (avg_distance, avg_distance_emp_tau, std_distance, std_distance_emp_tau)

    if epoch % num_discrim_updates == 0:
        """TRAIN GENERATOR"""
        optimiser_G.zero_grad()
        z = torch.randn(batch_size, latent_dim, dtype=torch.float32).to(device)
        gen_future_states = netG(z, current_states)
        errD_fake = netD(current_states, gen_future_states)
        loss_G_p1 = errD_fake.mean()
        if args.diversity_sensitivity:
            z_alt = torch.randn(batch_size, latent_dim, dtype=torch.float32).to(device)
            gen_future_states_alt = netG(z_alt, current_states).detach()
            if args.alt_ds_norm:
                norm1 = torch.norm(gen_future_states, p=1, dim=-1).detach()
                norm2 = torch.norm(gen_future_states_alt, p=1, dim=-1).detach()
                state_diff_l1 = F.l1_loss(gen_future_states/(norm1.view(-1,1)), gen_future_states_alt/(norm2.view(-1,1)), reduce=False).sum(dim=-1) / state_dim
            else:
                state_diff_l1 = F.l1_loss(gen_future_states, gen_future_states_alt, reduce=False).sum(dim=-1) / state_dim
            z_diff_l1 = F.l1_loss(z.detach(), z_alt.detach(), reduce=False).sum(dim=-1) / latent_dim
            ds_loss = -1*args.ds_loss_coeff*(state_diff_l1 / (z_diff_l1+1e-5)).mean()
            loss_G = loss_G_p1 + ds_loss
        else:
            loss_G = loss_G_p1
        if args.l2_reg_g:
            p_g = 0.0
            for p in netG.parameters():
                p_g += torch.dot(p.view(-1), p.view(-1))
            loss_G += args.l2regparam*p_g
        loss_G.backward()
        total_norm = 0
        for p in netG.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
        loss_record["generator_grad_norm"].append(total_norm**(1.0/2))
        loss_record["G"].append(loss_G_p1.item())
        str_to_print += "[G_loss (base): %f] " % loss_G_p1.item()
        if args.diversity_sensitivity:
            loss_record["DS"].append(ds_loss.item())
            loss_record["G_tot"].append(loss_G.item())
            str_to_print += "[DS loss: %f] [G_loss (tot): %f]" % (ds_loss.item(), loss_G.item())
        optimiser_G.step()

    print(str_to_print)
    if epoch % args.eval_every == 0:
        joblib.dump(loss_record, "results/nearby_states/"+experiment_name+"/losses.pkl")
        joblib.dump((netG.state_dict(), netD.state_dict()), "results/nearby_states/"+experiment_name+"/parameters.pkl")
        #generate_sample_states(6, 8, epoch)
        big_eval(500, 500)
