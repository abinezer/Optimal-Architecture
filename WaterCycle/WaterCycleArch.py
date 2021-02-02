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

def evaluate2(model, val_loader):
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
    
class Raindrop:
    def __init__(self,x0):
        self.position = x0
    
    def evaluate(self):
    	#self.err_i=costFunc(self.position_i)
        #net = Net(int(round(self.position_i[0])))
        net = WineModel(round(self.position))
    	#print(net)
        #optimizer = optim.Adam(net.parameters(), lr=0.05)
        #def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        #history = []
        lr = 1e-2
        epochs = 100
        optimizer = torch.optim.SGD(net.parameters(), lr)
        for epoch in range(epochs):
            # Training Phase 
            for batch in train_loader:
                loss = net.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # Validation phase
            result = evaluate2(net, val_loader)
            net.epoch_end(epoch, result, epochs)
            #history.append(result)  #appends total validation loss of whole validation set epoch wise
        print(result['val_loss'])
        return result['val_loss']

import random
from operator import itemgetter 
class WaterCycle:
    def __init__(self,bounds,Npop,dmax, max_it):
        global num_dimensions
        
        C = 2
        
        Nsr = 5
        Nrivers = Nsr - 1
        Nstreams = Npop-Nsr
        
        num_dimesnions = 1
        
        population = []
        for i in range(Npop):
            x0 = random.randint(bounds[0],bounds[1])
            print('Raindrop ',i,' is ', x0)
            population.append(Raindrop(x0))
            
        ListOfRaindrops = []
        print('Iteration 0')
        print('************************')
        for j in range(Npop):
            print('Raindrop ',j)
            err_j = population[j].evaluate()
            ListOfRaindrops.append([j,err_j])
        
        
        res = list(sorted(ListOfRaindrops, key = itemgetter(1), reverse = False)[:Nsr]) 
        print('ListOfRaindrops ', ListOfRaindrops)
        print('res' ,res)
        #print(res[0][0])
        
        totalCostOfNsr = 0
        for i in range(Nsr):
            totalCostOfNsr = totalCostOfNsr + res[i][1]
        
        NSn = []
        for i in range(Nsr):
            NSn.append([res[i][0],round((res[i][1]/totalCostOfNsr)*Nstreams)])
            print('NSn: ',NSn)
        
        NsrList = []
        for i in NSn:
            NsrList.append(i[0])
        print('NsrList: ',NsrList)


        NstreamsList = []
        for i in ListOfRaindrops:
            if i[0] not in NsrList:
                NstreamsList.append([i[0],i[1]])
        print('NstreamsList Unsorted: ', NstreamsList)
        NstreamsList = list(sorted(NstreamsList, key = itemgetter(1), reverse = False))
        print('NstreamsList Sorted: ',NstreamsList)
        
        streams = []
        for i in NstreamsList:
            streams.append(i[0])
        print('streams',streams)
        
        i = 1
        while(i < max_it):
            print('Iteration ',i)
            print('************************')
            
            counter = 0
            limit = 0
            for p in NSn:
                prevLimit = limit
                limit = limit + p[1]
                for j in range(prevLimit,limit):
                    if j <= len(NstreamsList):
                        drop = NstreamsList[j][0]
                        print('Raindrop ',drop)
                        population[drop].position = population[drop].position + (random.uniform(0,1) * C * (population[p[0]].position - population[drop].position))
                        if population[drop].position > bounds[1]:
                            population[drop].position = bounds[1]
                        elif population[drop].position < bounds[0]:
                            population[drop].position = bounds[0]
                        if population[drop].evaluate() >  population[p[0]].evaluate():   #stream <--> river<-->sea
                            print('Interchanging stream and river!!!!!')
                            NstreamsList[j][0] = p[0]       
                            res[counter][0] = drop
                            NSn[counter][0] = drop
                        
                counter+=1
            
            
            NsrList = []
            for c in NSn:
                NsrList.append(c[0])

            print('NsrList: ',NsrList)


            NstreamsList = list(sorted(NstreamsList, key = itemgetter(1), reverse = False))
            print('NstreamsList: ',NstreamsList)

            streams = []
            for k in NstreamsList:
                streams.append(k[0])
            print('streams',streams)

            for k in range(Npop):
                print('Raindrop ',k,' is ', population[k].position)
            
            
            for j in range(1,len(NSn)):
                river = NSn[j][0]
                sea = NSn[0][0]
                print('Raindrop ', river, ' which is a river ')
                population[river].position = population[river].position + (random.uniform(0,1) * C * (population[sea].position - population[river].position))
                if population[river].position > bounds[1]:
                    population[river].position = bounds[1]
                elif population[river].position < bounds[0]:
                    population[river].position = bounds[0]
                if population[river].evaluate() > population[sea].evaluate():
                    print('Interchanging river sand sea!!!!')
                    NSn[0][0] = river
                    NSn[j][0] = sea
                
            for j in range(1,len(NSn)):
                seaPos = NSn[0][1]
                riverPos = NSn[j][1]
                if abs(seaPos - riverPos) < dmax or random.uniform(0,1) < 0.1:
                    print('Evapouration Process for river!!')
                    population[NSn[j][0]].position = bounds[0] + (random.uniform(0,1)*(bounds[1]-bounds[0]))
            
                    
            dmax = dmax - (dmax/max_it)
            NsrList = []
            for g in NSn:
                NsrList.append(g[0])

            print('NsrList: ',NsrList)


            NstreamsList = list(sorted(NstreamsList, key = itemgetter(1), reverse = False))
            print('NstreamsList: ',NstreamsList)

            streams = []
            for t in NstreamsList:
                streams.append(t[0])
            print('streams',streams)
            
            
            i = i + 1
            
        print('Final')
        print(NSn[0][1])
#--- RUN ----------------------------------------------------------------------+

    #initial = 5               # initial starting location [x1,x2...]
bounds=[1,10]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
WaterCycle(bounds,Npop = 17,dmax = 0.01, max_it=2)



