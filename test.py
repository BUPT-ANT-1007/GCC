from model import *
import time
import os
import logging
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))

log_path = './logs/'
if not os.path.isdir(log_path):
    os.makedirs(log_path)
log_name = log_path + rq + '.log'
print(log_name)
logfile = log_name
if not os.path.isfile(logfile):
    os.system(r"touch {}".format(logfile))
fh = logging.FileHandler(logfile, mode='a')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info('Program start...LFAS...')

import sys
# dsModel = sys.argv[0]
dsName = sys.argv[1]



# Y = np.load('./testData/test/five/S_test_'+ dsName +'.npy')
Y = np.load('./testData/test/9/S_test_'+dsName+'.npy')

label = np.load('./testData/test/9/S_label_'+dsName+'.npy')
# label = np.load('./testData/test/'+dsName+'/S_label_'+dsName+'.npy')

in_feat, in_feat_1 = Y.shape
Y = torch.from_numpy(Y)
label = torch.from_numpy(label)
g2 = create_graph_ar(numOfNodes=81, pattern=2)

device = torch.device("cuda:5")
# model = torch.load('./modelSave/twoLinear/LFAS-GLL-'+dsName+'.pt', map_location='cuda:5')
model = torch.load('./modelSave/9/LFAS-GLL-town.pt', map_location='cuda:5')
# model = torch.load('./modelSave/star/LFAS-GLL-'+dsName+'_ori.pt', map_location='cuda:5')
model.eval()
re_Y = np.zeros((in_feat, in_feat_1))

# prediction = model(g1, g2, Y.to(device))
prediction = model(g2, Y.to(device))
re_Y = prediction.cpu().detach().numpy()

re_Y = restore_S(re_Y)
Y_GT = restore_S(label.cpu().detach().numpy())
re = np.zeros((9,9))
all_psnr = 0

for i in range(81):
    re[i//9, i%9] = psnr(re_Y[i], Y_GT[i])
    all_psnr += re[i//9, i%9]
    logger.info('image plane: {}, {} | PSNR: {}'.format(i//9+1, i%9 + 1, re[i//9, i%9]))
print("the average of PSNR:%d ", all_psnr/81)
np.save('./result/re_'+dsName+'',re_Y)

