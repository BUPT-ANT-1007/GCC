from model import *
import time
import os
import logging
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
dataName = 'town'


log_path = './logs/'
if not os.path.isdir(log_path):
    os.makedirs(log_path)
log_name = log_path + dataName + rq + '.log'
logfile = log_name
if not os.path.isfile(logfile):
    os.system(r"touch {}".format(logfile))
fh = logging.FileHandler(logfile, mode='a')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info('Program start...')


#Main
Y = np.load('./testData/test/9/S_test_'+ dataName +'.npy')
label = np.load('./testData/test/9/S_label_'+dataName+'.npy')  #load label
in_feat, in_feat_1 = Y.shape
Y = torch.from_numpy(Y)
label = torch.from_numpy(label)
g2 = create_graph_ar(numOfNodes=81, pattern=2)


import torch.optim as optim

device = torch.device("cuda:3")
model = LFARN_Z(in_feat,in_feat,in_feat).double()

model.to(device)

loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
model.train()
epoch_losses = []
logger.info('Training start...')
epoch_loss = 0
epoch_mark = 0

temp = float('inf')
for epoch in range(100000):
    prediction = model(g2, Y.to(device))
    loss = loss_func(prediction, label.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_loss = loss.detach().item()
    print('epoch = ',epoch , 'loss = ',epoch_loss)
    if loss<temp:
        temp = loss
        epoch_mark = epoch
        torch.save(model, './modelSave/9/LFAS-GLL-'+dataName+'.pt')
    logger.info('DataName {}, Epoch {}, loss {:.6f}, Mark {}, minLoss{}'.format(dataName, epoch, epoch_loss, epoch_mark, temp))