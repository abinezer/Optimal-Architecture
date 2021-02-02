import pandas as pd
df = pd.read_csv("archive/winequality-red.csv")
input_cols = list(df.columns)[:-1]
output_cols = ['quality']

def dataframe_to_arrays(df):
	# Make a copy of the original dataframe
	df1 = df.copy(deep=True)
	# Extract input & outupts as numpy arrays
	inputs_array = df1[input_cols].to_numpy()
	targets_array = df1[output_cols].to_numpy()
	return inputs_array, targets_array
    
inputs_array, targets_array = dataframe_to_arrays(df)
import torch
inputs = torch.Tensor(inputs_array)
targets = torch.Tensor(targets_array)
from torch.utils.data import DataLoader, TensorDataset, random_split
dataset = TensorDataset(inputs, targets)

num_rows = len(df)
val_percent = 0.8 # between 0.1 and 0.2
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size


train_df, val_df = random_split(dataset, [train_size, val_size])

batch_size = 50
train_loader = DataLoader(train_df, batch_size, shuffle=True)
val_loader = DataLoader(val_df, batch_size)

input_size = len(input_cols)
output_size = len(output_cols)

import torch.nn as nn
import torch.nn.functional as F
class WineModel(nn.Module):
    def __init__(self,number_of_neurons):
        super().__init__()     
        self.fc1 = nn.Linear(input_size, number_of_neurons)
        self.fc2 = nn.Linear(number_of_neurons,output_size)
        # fill this (hint: use input_size & output_size defined above)
        #model initialized with random weight
        
    def forward(self, xb):
        xb = F.relu(self.fc1(xb))
        out = self.fc2(xb)             # batch wise forwarding
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)         
        # Calcuate loss
        loss = F.l1_loss(out, targets)  # batch wise training step and loss
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss =F.l1_loss(out, targets)       # batch wise validation and loss    
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine val losses of all batches as average
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch+1) % 500 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)  #appends total validation loss of whole validation set epoch wise
    return history

class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

    
        self.velocity_i.append(random.uniform(-1,1))
        self.position_i.append(x0)

    # evaluate current fitness
    def evaluate(self):
    	#self.err_i=costFunc(self.position_i)
        #net = Net(int(round(self.position_i[0])))
        net = WineModel(round(self.position_i[0]))
    	#print(net)
        #optimizer = optim.Adam(net.parameters(), lr=0.05)
        #def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        #history = []
        lr = 1e-2
        epochs = 2000
        optimizer = torch.optim.SGD(net.parameters(), lr)
        for epoch in range(epochs):
            # Training Phase 
            for batch in train_loader:
                loss = net.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # Validation phase
            result = evaluate(net, val_loader)
            net.epoch_end(epoch, result, epochs)
            #history.append(result)  #appends total validation loss of whole validation set epoch wise
        self.err_i = result['val_loss']

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        r1=random.random()
        r2=random.random()

        vel_cognitive=c1*r1*(self.pos_best_i[0]-self.position_i[0])
        vel_social=c2*r2*(pos_best_g[0]-self.position_i[0])
        self.velocity_i[0]=w*self.velocity_i[0]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        self.position_i[0]=self.position_i[0]+self.velocity_i[0]

        # adjust maximum position if necessary
        if self.position_i[0]>bounds[1]:
        	self.position_i[0]=bounds[1]

        # adjust minimum position if neseccary
        if self.position_i[0] < bounds[0]:
        	self.position_i[0]=bounds[0]


import math
import random

class PSO():
    def __init__(self,bounds,num_particles,maxiter):
        global num_dimensions

        num_dimensions= 1
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            x0 = random.randint(bounds[0],bounds[1])
            print('Particle ',i,' position is ', x0)
            swarm.append(Particle(x0))
        #print('Swarm : ',swarm)

        # begin optimization loop
        i=0
        while i < maxiter:
            print('Iteration :', i)
            print('*****************************')
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                print('Particle: ',j)
                swarm[j].evaluate()
                #print('swarm[{}].evaluate = {}'.format(j, swarm[j].evaluate(costFunc)))

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)
                print('pos_best_g so far', pos_best_g)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        # print final results
        print('FINAL:')
        print(pos_best_g)
        print(err_best_g)

if __name__ == "__main__":

#--- RUN ----------------------------------------------------------------------+

    #initial = 5               # initial starting location [x1,x2...]
    bounds=[1,10]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
    PSO(bounds,num_particles=7,maxiter=10)
    
  
