import torch.cuda
import torch
import numpy as np
import sys
import torch.nn as nn
import time
from sklearn.preprocessing import FunctionTransformer
import wandb
import torch.nn.functional as F

#-------------------------------------------------------------------------
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.set_printoptions(threshold=sys.maxsize) 
torch.set_printoptions(threshold=10_000)
#-------------------------------------------------------------------------


ga = np.load("TestingGammaNew.npy", allow_pickle=True)
no = np.load("TestingnurtronNew.npy", allow_pickle=True)


transformer = FunctionTransformer(np.log1p, validate=True)
ga = transformer.transform(ga)
no = transformer.transform(no)



ga = torch.Tensor(ga).cuda()
no = torch.Tensor(no).cuda()
input_data = torch.Tensor(np.load("Input_new.npy", allow_pickle=True)).cuda()
predict_data = torch.Tensor(np.load("label_new.npy", allow_pickle=True)).cuda()
input_data = input_data.type(torch.FloatTensor)
predict_data = predict_data.type(torch.LongTensor)

top20= np.float64([99.9]) 
gacount = 0
nocount = 0
correctCount =0 
wrongCount =0
input_size = 248
wrongCount2 =0
counter1 = 0
counter2 = 0
best = np.float64([99]) #antioverfit   
a = np.float64([99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99]) #antioverfit 
BATCH_SIZE = 100
num_epochs = 5000000
print_interval = 3000 
testing_loss = 0.0
counter = 0
DROPOUT1 = .3
DROPOUT2 = .1
DROPOUT3 = .3
num_layers1 = 1
num_layers2 = 1 
num_layers3 = 2
hidden_size = 100 
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size,  num_layers1, num_layers2, num_layers3, num_classes):
        super(RNN, self).__init__()
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2
        self.num_layers3 = num_layers3
        self.hidden_size = hidden_size 

        self.drop1 = torch.nn.Dropout(DROPOUT1)  
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers1, batch_first=True , )

        self.drop2 = torch.nn.Dropout(DROPOUT2) 
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers2, batch_first=True , )

        self.drop3 = torch.nn.Dropout(DROPOUT3) 
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers3, batch_first=True , )

        self.fc = nn.Linear(hidden_size, num_classes)                                                
        
    def forward(self, x):
        h01 = torch.zeros(self.num_layers1, x.size(0), self.hidden_size).to(device) 
        c01 = torch.zeros(self.num_layers1, x.size(0), self.hidden_size).to(device) 

        h02 = torch.zeros(self.num_layers2, x.size(0), self.hidden_size).to(device) 
        c02 = torch.zeros(self.num_layers2, x.size(0), self.hidden_size).to(device) 

        h03 = torch.zeros(self.num_layers3, x.size(0), self.hidden_size).to(device) 
        c03 = torch.zeros(self.num_layers3, x.size(0), self.hidden_size).to(device) 
      
        out, _ = self.lstm1(x, (h01,c01)) 
        out = self.drop1(out)

        out, _ = self.lstm2(out, (h02,c02)) 
        out = self.drop2(out)

        out, _ = self.lstm3(out, (h03,c03)) 
        out = self.drop3(out)

        out = out[:, -1, :]
        out = self.fc(out)                
        return out

input_size = 248
num_classes = 2


lr  = [ 0.0005, 0.001, .009]
betasright = [ 0.998, 0.997, .996]
betasleft = [0.6, 0.5, 0.4]
wd  = [1e-10, 1e-11, 1e-12]
eps = [1e-06, 1e-07, 1e-08]

config_dict = {
    "lr"  : [ 0.0005, 0.001, .009],
    "betasright" : [ 0.998, 0.997, .996],
    "betasleft" : [0.6, 0.5, 0.4],
    "wd"  :  [1e-10, 1e-11, 1e-12],
    "eps" : [1e-06, 1e-07, 1e-08]
}




model = RNN(input_size, hidden_size, num_layers1, num_layers2, num_layers3 , num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RAdam(model.parameters(), lr=0.001, betas=(0.6, 0.997), eps=1e-06, weight_decay=1e-11 )

model.train()
model.to(device)
input_data.to(device)
predict_data.to(device)

PATH = "model.pt"
torch.save({'model_state_dict1': model.state_dict(),}, PATH)
start_time = time.time()
overallacc = 0


for WD in wd:
    for EPS in eps:
        for LR in lr:
            for BETASRIGHT in betasright:
                for BETASLEFT in betasleft:
                    counter = counter + 1

                    checkpoint = torch.load(PATH)
                    model.load_state_dict(checkpoint['model_state_dict1'])

                    optimizer = torch.optim.RAdam(model.parameters(), lr=LR, betas=(BETASLEFT, BETASRIGHT), eps=EPS, weight_decay=WD  )
                    a = np.float64([99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99]) #antioverfit     
                    testing_loss = 0.0
                    model.train().cuda()

                    
                    #wandb.init(project="my-test-project", entity="samuelfipps" , config=optimizer)

                    for epoch in range(num_epochs):
                        start_time = time.time()

                        if(overallacc < a[15]): # part of anti overfit
                            train_loss = 0.0        
                            testing_loss = 0.0

                            model.train()
                            for i in (range(0, len(input_data), BATCH_SIZE)):
                                batch_X = input_data[i:i+BATCH_SIZE]
                                batch_y = predict_data[i:i+BATCH_SIZE]

                                batch_X = batch_X.to(device) #gpu                        # input data here!!!!!!!!!!!!!!!!!!!!!!!!!!
                                batch_y = batch_y.to(device) #gpu                    # larget data here!!!!!!!!!!!!!!!!!!!!!!!!!!

                                batch_X = batch_X.reshape(-1, 1, input_size).to(device)
                                output = model(batch_X)
                                
                                loss = criterion(output, batch_y).to(device)
                                #wandb.log({"loss": loss})

                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                
                            
                            print(f"Epoch [{epoch + 1}/{num_epochs}], " f"Step [{i + 1}/{len(input_data)}], " f"Loss: {loss.item():.4f}")
                            secondTime = time. time()
                            print("total time for 1 epoch: ", secondTime-start_time)

                            model.eval()
                            wrongCount = 0
                            counter1 = 0
                            rightcounter = 0
                            with torch.no_grad():

                                for i in (range(0, len(ga), BATCH_SIZE)):

                                    batch_X = ga[i:i+BATCH_SIZE]
                                    batch_X = batch_X.reshape(-1, 1, input_size).to(device)
                                    output = model(batch_X)

                                    _, pred = torch.max(output, dim=1)
                                    for i in pred:
                                        counter1 = counter1 + 1
                                        if i !=0:
                                            wrongCount = wrongCount + 1
                                        else:
                                            rightcounter = rightcounter + 1

                            #neutron pulse tester
                            model.eval()
                            wrongCount2 = 0
                            counter2 = 0
                            rightcounter2 =0
                            with torch.no_grad():
                                
                                for i in (range(0, len(no), BATCH_SIZE)):

                                    batch_X = no[i:i+BATCH_SIZE]
                                    batch_X = batch_X.reshape(-1, 1, input_size).to(device)
                                    output = model(batch_X)

                                    _, pred = torch.max(output, dim=1)
                                    for i in pred:
                                        counter2 = counter2 + 1
                                        if i ==0:
                                            wrongCount2 = wrongCount2 + 1
                                        else:
                                            rightcounter2 = rightcounter2 + 1
                            print()
                            print(f"Accuracy for No pluse: {wrongCount2 / (counter2) * 100:.4f}%")
                            print(f"Accuracy for ga pluse: {wrongCount / (counter1) * 100:.4f}%")

                            print("Wrong count for No pluse: ", wrongCount2)
                            print("Wrong count for ga pluse: ", wrongCount)

                            print("Right count for No pluse: ", rightcounter)
                            print("Right count for ga pluse: ", rightcounter2)
                            
                            accuracy2 = wrongCount2 
                            accuracy = wrongCount 

                            wrongcountoverall = wrongCount2+ wrongCount
                            overallacc = (accuracy2+accuracy) /(counter1 + counter2)
                            #wandb.log({"Testing acc": overallacc*100})
                            #wandb.log({"Wrong count Overall": wrongcountoverall})

                            print(f"Accuracy overall : {overallacc *100:.4f}%")
                            print("Wrong count overall: ", wrongcountoverall)

                            trainttime = time. time()
                            print("total time for testing: ", trainttime-start_time)
                            print()

                            overallacc = overallacc*100
                            a = np.insert(a,0,overallacc) # part of anti overfit         
                            a = np.delete(a,22) 

                            #print("top20 list = "  ,top20)
                            if epoch == 0 and counter == 1:  
                                top20[0] = overallacc

                            if epoch == 0 and counter != 1:
                                top20 = np.append(top20, overallacc) 

                            elif overallacc < top20[counter-1]:              
                                top20[counter-1] = overallacc
                            
                    #wandb.finish(exit_code=0)
                    torch.save(model, "models/GRUModel.pth")
                    print(optimizer)
                    print("lr= ", LR, "betaright= ", BETASRIGHT, "betaleft= ", BETASLEFT, " wd= ", WD, "eps= ", EPS)
                    print("round: ", counter, " out of 243")
                    best = np.append(best, overallacc)
print(best)
                    
print("the top ones are: ")
print(top20)


#average = np.average(top20) 
#print("Average = ")
#print(average*100)
                    





