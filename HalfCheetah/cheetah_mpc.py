import numpy as np
from casadi import *
from scipy import linalg
import gymnasium as gym
from gymnasium.utils.save_video import save_video
import time
import matplotlib.pyplot as plt
#from AG_gym_nn import normalize_A, cycle_dynamics_pontry_train_A, cycle_dynamics_pontry_train_B,cycle_dynamics_nn_diff_A
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
import torch
import os
import wandb
import torch.optim as optim
import torch.distributions as dist
import random


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dev_num=3#for cupy
torch.manual_seed(32)

def rand_with_seed(M, seed):
    np.random.seed(seed)
    return np.random.randint(0, 100,M)
# set_seed(0)

class WandB:
    os.environ["WANDB_API_KEY"] = "a0fc75f04fa27bc24039cf264e6500367853626f"
    project_name = "ad_cheetah"

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_cost = float('inf')
        self.early_stop = False

    def __call__(self, cost):
        if cost < self.best_cost - self.min_delta:
            self.best_cost = cost
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class new_cheetah(HalfCheetahEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset_model(self):

        qpos = self.init_qpos + np.array([0.05479121, -0.01222431, 0.07171958, 0.03947361, -0.08116453, 0.09512447, 0.05222794, 0.05721286, -0.07437727])
        qvel = (
            self.init_qvel
            + np.array([ -0.08530439, 0.0879398, 0.07777919, 0.00660307, 0.11272412, 0.04675093, -0.08592925, 0.03687508, -0.09588826])
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
    
def analyze_noise(noise, name):
    if isinstance(noise, list):
        noise = torch.stack(noise)
    
    print(f"Analysis for {name}:")
    print(f"Shape: {noise.shape}")
    print(f"Min: {noise.min().item():.4f}, Max: {noise.max().item():.4f}")
    print(f"Mean: {noise.mean().item():.4f}, Std: {noise.std().item():.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.hist(noise.flatten().numpy(), bins=50, density=True)
    plt.title(f"Distribution of values for {name}")
    plt.show()

class MPC:
    
    def __init__(self,  policy_network, horizon, learning_rate,desired_states,M,nx,nu,lambda_l1=0.001):
        
        self.policy_network = policy_network
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate,weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=50, verbose=True)
        self.desired_states = torch.tensor(desired_states, dtype=torch.float32, device="cuda")
        self.Q = torch.tensor(1*np.diag([ 100, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0]), dtype=torch.float32) 
        self.R=torch.tensor(0.001*np.diag([1,1,1,1,1,1]), dtype=torch.float32) 
        self.Qn = torch.tensor(10*np.diag([ 100, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0]), dtype=torch.float32) 
        self.M=M
        self.nx=nx
        self.nu=nu
        self.lambda_l1 = lambda_l1
        self.seed=0
        self.current_seed_index = 0
        self.seed_sequence=rand_with_seed(1000,seed=32)
        self.sym_solver()
        # set up wandb
        os.environ["WANDB_API_KEY"] = "a0fc75f04fa27bc24039cf264e6500367853626f"
        wandb.init(project=WandB.project_name)
        # wandb.watch(self.policy_network, log='gradients',log_freq=50)

    def sym_solver(self):
        self.A = SX.sym('A', self.nx, self.nx)
        self.B = SX.sym('B', self.nx, self.nu)

        for i in range(self.horizon-1):
            self.loss_function = 0
            self.symbolic_data = []
            for j in range(self.M):
                x_t_sym = SX.sym(f'x_t_{j}_{i}', self.nx, 1)
                u_t_sym = SX.sym(f'u_t_{j}_{i}', self.nu, 1)
                x_tp1_sym = SX.sym(f'x_tp1_{j}_{i}', self.nx, 1)
                self.symbolic_data.append(x_t_sym)
                self.symbolic_data.append(u_t_sym)
                self.symbolic_data.append(x_tp1_sym)
                self.loss_function += sumsqr(self.A @ x_t_sym + self.B @ u_t_sym - x_tp1_sym)
        self.qp_param = vertcat(*self.symbolic_data)
        self.qp_var = vertcat(reshape(self.A, -1, 1), reshape(self.B, -1, 1))
        self.qp_cost = self.loss_function
        self.qp_program = {'x': self.qp_var, 'f': self.qp_cost, 'p': self.qp_param}
        opts = {'print_time': 0, 'osqp': {'verbose': False}}
        self.qp_solver_fn_ = qpsol('qp_solver', 'osqp', self.qp_program, opts)
    
    def get_next_state(self,current_state, action):
        action_np = action.detach().cpu().numpy()
        device = action.device
        env = new_cheetah(render_mode=None, exclude_current_positions_from_observation=False)
        env.reset(seed=self.seed)
        initial_state=current_state
        #print("iinitial states:",initial_state)
        env.set_state(initial_state[:9],initial_state[9:])
       # print("action shapppe:",action.shape)
        observation, _, _, _, info = env.step(action_np)
        next_state = torch.tensor(observation, dtype=torch.float32, device=device)
        env.close()
        #print("next states shape:",next_state)
        return next_state

    def actseq_next_states(self,current_state,action_seq):
        actions_np = action_seq.detach().cpu().numpy()
        device = action_seq.device
        env = new_cheetah(render_mode=None, exclude_current_positions_from_observation=False)
        env.reset(seed=self.seed)
        initial_state=current_state
        env.set_state(initial_state[:9],initial_state[9:])
        next_states=[]
        for i in range(action_seq.shape[0]):

            observation, _, _, _, info = env.step(actions_np[i])
            next_state = torch.tensor(observation, dtype=torch.float32, device=device)
            next_states.append(next_state)
        next_states=torch.stack(next_states,dim=0)

        return next_states
    
    def cal_AB(self, states, actions):
        seed = self.seed_sequence[self.current_seed_index]
        set_seed(seed)
        # Update to the next seed for the next call
        self.current_seed_index = (self.current_seed_index + 1) % len(self.seed_sequence)
        print(f"Seed set to: {seed}") 
        #print("states shape he:",states.shape)#51,18
        device = states.device
        ep = torch.FloatTensor(1).uniform_(-0.1, 0.1)
        # rng = np.random.default_rng(42)
        z1_matrices = [torch.tensor(truncated_gaussian((states.shape[0], self.nu)), dtype=torch.float32, device='cpu') for _ in range(self.M)]
        # z1_matrices = torch.normal(mean=0.0, std=1, size=(self.M, states.shape[0], self.nu)).to(device)#random noise
        # z1_matrices = torch.clamp(torch.normal(mean=0.0, std=1 ,size=(self.M, states.shape[0], self.nu)), min=-1, max=1).to(device)
        # analyze_noise(z2_matrices, "pre Truncated Gaussian") 
        # analyze_noise(z1_matrices, "Truncated Gaussian")
        # print("z1 matrixed",z1_matrices)
        u_new = torch.zeros((self.M, states.shape[0],self.nu), device=device)
        x_new = torch.zeros((self.M,  states.shape[0],self.nx), device=device)
        del_x = torch.zeros((self.M,  states.shape[0],self.nx), device=device)
        del_u = torch.zeros((self.M,  states.shape[0],self.nu), device=device)
        vec_AB = vertcat(reshape(self.A, -1, 1), reshape(self.B, -1, 1))

        AB_mapping_fn = Function('AB_Mapping_fn', [vec_AB], [self.A, self.B])
        # print("actions shape", actions.shape)
        # print("z1 shape:", z1_matrices[0].shape)
        
        for j in range(self.M):
            z1 = z1_matrices[j]
            #print("z1 shape:",z1.shape)
            u_new[j] = actions +  ep*z1
            u_new[j] = torch.clamp(u_new[j], min=-1.0, max=1.0)#u range -1 to 1
            # print("u s value:",u_new[j])
            with torch.no_grad():  # 
                x_new[j] = self.actseq_next_states(states[0],u_new[j])#initiali state,pertubed action
                del_u[j] = u_new[j] - actions
                del_x[j] = x_new[j] - states 
        
        A_fin = np.zeros([states.shape[0], self.nx, self.nx])
        B_fin = np.zeros([states.shape[0], self.nx, self.nu])
        
        for i in range(states.shape[0]):
            data = []
            for j in range(self.M):
                x_t = del_x[j, i-1,:]
                u_t = del_u[j, i-1,:]
                x_tp1 = del_x[j, i ,:]
                data.append(x_t)
                data.append(u_t)
                data.append(x_tp1)
        
            qp_param_value = torch.cat([item.flatten() for item in data])
            qp_param_value_np = qp_param_value.detach().cpu().numpy()
            
            sol = self.qp_solver_fn_(p=qp_param_value_np)
            sol_np = np.array(sol['x'])
          
            A= AB_mapping_fn(sol_np)[0]
            B= AB_mapping_fn(sol_np)[1]
            A_fin[i] = A
            B_fin[i] = B
        #print("A fin shape:",A_fin[49].shape) #A fin 51 elements 18*18 list
        A_fin=torch.tensor(A_fin, dtype=torch.float32, device=device)
        B_fin=torch.tensor(B_fin, dtype=torch.float32, device=device)
        return A_fin, B_fin

    
    def cost_function(self, states, actions):
        device = states.device
        cost = torch.tensor(0.0, dtype=torch.float32, device=device)
        desired_states = self.desired_states.to(device)
        Q = self.Q.to(device)
        R = self.R.to(device)
        
        # Calculate the state and action cost
        for i in range(states.shape[0]):
            state_deviation = states[i, :] - desired_states
            state_cost = 0.5 * (state_deviation @ Q @ state_deviation.T)
            action_cost = 0.5 * (actions[i,:] @ R @ actions[i,:].T)
            cost = cost + state_cost + action_cost
        
        # Add the L1 regularization term
        l1_reg = 0
        for param in self.policy_network.parameters():
            l1_reg += torch.sum(torch.abs(param))
        
        cost += self.lambda_l1 * l1_reg
        
        return cost
    
    # terminal cost function
    def terminal_cost(self, final_state):
        
        device = final_state.device
        desired_final_state = self.desired_states.to(device)
        Qn = self.Qn.to(device)  
        state_deviation = final_state - desired_final_state
        cost = 0.5 * (state_deviation @ Qn @ state_deviation.T)
        
        return cost
    
    def forward_pass(self, state):
        current_state = torch.tensor(state, dtype=torch.float32)
        next_states=[]
        actions=[]
        for i in range(self.horizon):
            action = self.policy_network.forward(current_state)
            next_state = self.get_next_state(current_state,action)
            next_states.append(next_state)
            actions.append(action)
        next_states=torch.stack(next_states,dim=0)
        actions=torch.stack(actions,dim=0)
        # print("next shape :,",next_states.shape)
        # print("nactions shape:",actions.shape)
        cost = self.cost_function(next_states, actions) + self.terminal_cost(next_states[-1])
        return next_states, actions, cost
#----------------------------------------gradients calculation--------------------------------------------------------------------------
        #c/x,c/u
    def compute_cost_gradients(self, states, actions):
        states = states.detach().requires_grad_(True)
        actions = actions.detach().requires_grad_(True)

        immediate_cost = self.cost_function(states, actions)
        # print(f"States shape: {states.shape}, requires_grad: {states.requires_grad}")
        # print(f"Actions shape: {actions.shacost_functionpe}, requires_grad: {actions.requires_grad}")

        #immediate_cost = self.cost_function(states, actions)
        # print(f"Immediate cost: {immediate_cost.item()}, requires_grad: {immediate_cost.requires_grad}")

        dc_dx_t = torch.autograd.grad(immediate_cost, states, create_graph=True, retain_graph=True, allow_unused=True)[0]
        dc_du_t = torch.autograd.grad(immediate_cost, actions, create_graph=True, retain_graph=True, allow_unused=True)[0]

        #print(f"dc_dx_t shape: {dc_dx_t.shape if dc_dx_t is not None else None}")
        # print(f"dc_du_t shape: {dc_du_t.shape if dc_du_t is not None else None}")

        if dc_dx_t is None:
            print("Gradient for states is None. This indicates a problem with the autograd graph.")
            dc_dx_t = torch.zeros_like(states)
        else:
            dc_dx_t = torch.where(dc_dx_t.isnan(), torch.zeros_like(dc_dx_t), dc_dx_t)

        if dc_du_t is None:
            print("Gradient for actions is None. This indicates a problem with the autograd graph.")
            dc_du_t = torch.zeros_like(actions)
        else:
            dc_du_t = torch.where(dc_du_t.isnan(), torch.zeros_like(dc_du_t), dc_du_t)
        # print("dcdx shape:",dc_dx_t.shape)
        return dc_dx_t, dc_du_t
    
    def compute_dxdtheta(self, states,actions):
        # states = normalize_tensor(states)
        states = states.detach().requires_grad_(True)
       
        # actions = normalize_tensor(actions)
        dfdx,dfdu = self.cal_AB(states,actions) # 51,18,18
        # dfdx = normalize_tensor(dfdx)
        # dfdu = normalize_tensor(dfdu)
        #print("A shape:",dfdx.shape)
        
        dudtheta=[]
        param_sizes = []

        #cal dudtheta
        for j in range(actions.shape[0]):
            dudtheta_t=[]
            for i in range(actions.size(1)):  # for each action dimension
                duidtheta = torch.autograd.grad(
                    outputs=actions[:-1].sum(),
                    inputs=self.policy_network.parameters(),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )
                param_sizes.append([tuple(g.size()) for g in duidtheta if g is not None])
                duidtheta_tensor = torch.cat([g.view(-1) for g in duidtheta if g is not None])
                dudtheta_t.append(duidtheta_tensor)#shape 9926 tensor
                #print("dud dthat shape:",dudtheta[0].shape)
            
            dudtheta_t = torch.stack(dudtheta_t, dim=0)#dudthat shape: torch.Size([6, 9926]),dudtheta done
            
            #print("dudthat shape:",dudtheta[1])
            dudtheta.append(dudtheta_t)
        dudtheta=torch.stack(dudtheta,dim=0)
        dudtheta=normalize_tensor(dudtheta)#torch size(51,6,9926)
        # print("dudthat shape:",dudtheta.shape)


        #cal terminal dudthat
        dudthetaT=[]
        for j in range(actions.shape[0]):
            dudthetaTt=[]
            for i in range(actions.size(1)): 
                dudthetaTi=torch.autograd.grad(
                        outputs=actions[-1].sum(),
                        inputs=self.policy_network.parameters(),
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True
                    )
                dudthetaTtensor=torch.cat([g.view(-1) for g in dudthetaTi if g is not None])
                dudthetaTt.append(dudthetaTtensor)
            dudthetaTt=torch.stack(dudthetaTt)
            dudthetaT.append(dudthetaTt)
        dudthetaT=torch.stack(dudthetaT,dim=0)
        dudthetaT=normalize_tensor(dudthetaT)
       # print("dudthat TTshape:",dudthetaT.shape)#(51,6,9926)

        # cal terminal dxdtheta
        dxdthetaT=torch.zeros(states.shape[1],dudtheta.shape[2])
        dxdthetaT=normalize_tensor(dfdx[-1]@dxdthetaT+dfdu[-1]@dudthetaT[-1])#torch.Size([18, 10310])
        #print("dx dthetaT shape:",dxdthetaT.shape)

        dxdtheta_t=torch.zeros(states.shape[1],dudtheta.shape[2])
        dxdtheta=[]
        for i in range(actions.size(0)):
            dxdtheta.append(dxdtheta_t)
            dxdtheta_t=dfdx[i-1]@dxdtheta_t+dfdu[i-1]@dudtheta[i-1]
        dxdtheta=torch.stack(dxdtheta,dim=0)
        dxdtheta=dxdtheta[:-1]
        #print("dxdthesta shape:",dxdtheta.shape)
        dxdtheta = normalize_tensor(dxdtheta)#([50, 18, 9926)
        return dxdtheta,dudtheta,param_sizes,dxdthetaT

    # dV_T/dx ,final cost's 
    def compute_terminal_cost_gradient(self, terminal_state):
        terminal_state = terminal_state.detach().requires_grad_(True)
        normalized_terminal_state = normalize_tensor(terminal_state)
        term_cost = self.terminal_cost(normalized_terminal_state)
    
        dV_dx_T = torch.autograd.grad(term_cost, terminal_state, create_graph=True, retain_graph=True)[0]
        
        if dV_dx_T is None:
            print("Gradient for terminal state is None. This indicates a problem with the autograd graph.")
            dV_dx_T = torch.zeros_like(terminal_state)
        else:
            dV_dx_T = torch.where(dV_dx_T.isnan(), torch.zeros_like(dV_dx_T), dV_dx_T)
        
        dV_dx_T = normalize_tensor(dV_dx_T)
        
        return dV_dx_T

    # compute total_grad todo:cant normalize
    def compute_gradients(self,states,actions):
        # actions = self.policy_network(states)  # [horizon, action_dim]
        dc_dx_t, dc_du_t = self.compute_cost_gradients(states, actions)#shape,51,18 and 51,6
        dc_dx_t = normalize_tensor(dc_dx_t[:-1])
        dc_du_t = normalize_tensor(dc_du_t[:-1])
        dxdtheta,dudtheta,param_sizes,dxdthetaT=self.compute_dxdtheta(states,actions)
        #compute terminal term
        dc_dx_T=self.compute_terminal_cost_gradient(states[-1,:])
        # dxTdtheta,_,_=self.compute_dxdtheta(states[-1].unsqueeze(0))
        djdthetaT=dc_dx_T@dxdthetaT#9926
        #print("dV dx T shappp:",dV_dx_T.shape)
        djdtheta=[]
        for i in range(states.shape[0]-1):
            djdtheta_t=dc_dx_t[i]@dxdtheta[i]+dc_du_t[i]@dudtheta[i]#shape 51,9926 dxdtheta 51 timesteps
            djdtheta.append(djdtheta_t)
        djdtheta=torch.stack(djdtheta,dim=0)#50,9926
        
        djdtheta=torch.sum(djdtheta,dim=0)#sum all
        djdtheta=djdtheta+djdthetaT
        #print("djdtheta shape:",djdtheta.shape)#final grad
        djdtheta = normalize_tensor(djdtheta)

        return djdtheta,param_sizes
#---------------------------update policy----------------------------------------------------------------------------------------
    
    def update_policy(self, gradients, param_sizes):
        # print("check shapes:",gradients.shape)#250,param_num
        
        restored_grads = self.restore_original_shapes([gradients], [param_sizes])[0]
        # print("resotre shape:",len(restored_grads))#8
        # print("resotre shape:",restored_grads[0].shape)#128,18
        # print("resotre shape:",restored_grads[1].shape)#128
        # print("resotre shape:",restored_grads[2].shape)#128
        # print("resotre shape:",restored_grads[3].shape)#128
        # print("resotre shape:",restored_grads[4].shape)#128
        # print("resotre shape:",restored_grads[5].shape)#128
        # print("resotre shape:",restored_grads[6].shape)#128
        # resotre shape: 8
        # resotre shape: torch.Size([128, 18])
        # resotre shape: torch.Size([128])
        # resotre shape: torch.Size([128, 128])
        # resotre shape: torch.Size([128])
        # resotre shape: torch.Size([128, 128])
        # resotre shape: torch.Size([128])
        # resotre shape: torch.Size([6, 128])

        self.optimizer.zero_grad()   
        for param, grad in zip(self.policy_network.parameters(), restored_grads):
            if param.size() == grad.size():
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    print(f"NaN or Inf gradient detected for parameter of size {param.size()}")
                    param.grad = torch.zeros_like(param)
                else:
                    param.grad = normalize_tensor(grad)
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return self.get_current_lr()

    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def restore_original_shapes(self, flattened_dudtheta, original_shapes):
        restored_dudtheta = []
        for flat_tensor, shapes in zip(flattened_dudtheta, original_shapes[0]):
            start = 0
            restored_params = []
            for shape in shapes:
                num_elements = np.prod(shape) 
                restored_params.append(flat_tensor[start:start+num_elements].view(shape))
                start += num_elements
            restored_dudtheta.append(restored_params)
        return restored_dudtheta
        

    def optimize_policy(self, initial_state, max_iterations=2000, convergence_threshold=1e-4):
        current_state = initial_state
        costs = [] 
        # early_stopping = EarlyStopping(patience=50, min_delta=0.001)
        best_cost=float('inf')
      
        for iteration in range(max_iterations):
            # forward
            states, actions, total_cost = self.forward_pass(current_state)
            if isinstance(total_cost, torch.Tensor):
                total_cost = total_cost.detach().cpu().item()
            
            costs.append(total_cost)  
            print(f"epochs {iteration}: Cost = {total_cost}")
            states=torch.tensor(states, dtype=torch.float32)    
            gradients,param_sizes = self.compute_gradients(states,actions)

            current_lr = self.update_policy(gradients, param_sizes)
            # self.scheduler.step(total_cost)  #update the learning rate based on the cost
            
            wandb.log({
                "Cost": total_cost,
                "Learning Rate": current_lr
            })
                
            if total_cost<best_cost:
                best_cost=total_cost
                torch.save(self.policy_network.state_dict(),'best_model.pth')
                print("best model saved")

            # early_stopping(total_cost)
            # if early_stopping.early_stop:
                
            #     print(f"Early stopping triggered at epoch {iteration}")
            #     break    

            # save model every 200 itr
            if (iteration + 1) % 25 == 0:
                self.save_model(iteration+1)
                
            # check convergence
            if total_cost < convergence_threshold:
                print(f"Converged at epoch {iteration} with cost {total_cost}")
                break

        self.save_model(iteration+1)
        
        # plot 
        plt.figure(figsize=(10, 5))
        plt.plot(costs, marker='o', linestyle='-', color='b')
        plt.title('Cost /itr')
        plt.xlabel('epochs')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.savefig('cost_plot.png')  
        #plt.show()

        wandb.log({"Cost Plot": wandb.Image("cost_plot.png")})
        return self.policy_network
    
    def save_model(self, itr_num):
        torch.save(self.policy_network.state_dict(), f'model_{itr_num}.pth')
        print("model saved")

#----------------------------------------------------------utils---------------------------------------------------------------------
def normalize_tensor(tensor, dim=0):
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True)
    return (tensor - mean) / (std + 1e-8)

def simulation(actions):
    actions=actions.numpy()
    env = new_cheetah(render_mode="human",exclude_current_positions_from_observation=False)
    observation, info = env.reset(seed)
    #print("Initial Observation:", observation)
    print("u shape[1] :",actions.shape)
    for i in range(actions.shape[0]):
        action = actions[i,:]*1 
        #print("action:",action)
        #action = np.clip(action, -1, 1)
        observation, reward, _, _, info = env.step(action)
        #print("obser :",observation.shape)

        env.render()
    env.close()

def animation(actions, video_path="videos", fps=20):
    actions=actions.numpy()
    env = new_cheetah(render_mode='rgb_array',exclude_current_positions_from_observation=False)
    observation, info = env.reset(seed)

    step_starting_index = 0
    episode_index = 0
    frames = []

    for i in range(actions.shape[0]):
        action = actions[i, :] 
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())

        if terminated or truncated:
            save_video(
                frames,
                video_path,
                fps=fps,
                step_starting_index=step_starting_index,
                episode_index=episode_index
            )
            step_starting_index = i + 1
            episode_index += 1
            frames = []  
            env.reset()

    if frames:
        save_video(
            frames,
            video_path,
            fps=fps,
            step_starting_index=step_starting_index,
            episode_index=episode_index
        )

    env.close() 

def truncated_gaussian(shape, mean=0, std=1, min_val=-1, max_val=1):
    # np.random.seed(42)
    values = np.random.normal(loc=mean, scale=std, size=shape)
    values = np.clip(values, min_val, max_val)
    return values


wandb.finish()