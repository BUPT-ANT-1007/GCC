import numpy as np
import torch
import dgl
import math

if __name__ == '__main__':
    # import cv2
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def gcn_message_func(edges):
    return {'msg': edges.src['h']}


def gcn_reduce_func(nodes):
    accum = torch.mean(nodes.mailbox['msg'], 1)
    return {'h': accum}


def psnr(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def creat_graph_sr(num_nodes=169):
    G = dgl.DGLGraph()
    G.add_nodes(num_nodes)
    p1 = [-1, 1]
    for i in range(num_nodes):
        if i in [0, 12, 168, 156]:
            continue
        x = i // 13
        y = i % 13
        for m in p1:
            for n in p1:
                new_x = x + m
                new_y = y + n
                if 0 <= new_x <= 12 and 0 <= new_y <= 12:
                    new_i = new_x * 13 + new_y
                    if new_i not in [0, 12, 168, 156]:
                        G.add_edge(new_i, i)
    return G


def create_graph_ar(numOfNodes=169, pattern=5):  #It needs to be modified according to the dimension of the light field image
    G = dgl.DGLGraph()
    G.add_nodes(numOfNodes)
    size = int(numOfNodes ** 0.5)
    set = np.linspace(-pattern + 1, pattern - 1, num=2 * pattern - 1)
    sampleMatrix = samplePattern(numOfNodes,pattern)
    # sampleMatrix = samplePattern(numOfNodes)
    likely = np.diag(sampleMatrix).reshape(size, size)
    for i in range(size):
        for j in range(size):
            if likely[i, j] == 1:
                G.add_edges(9 * i + j, 9 * i + j)  #It needs to be modified according to the dimension of the light field image
                continue
            for block_i in set:
                block_i = int(block_i)
                for block_j in set:
                    block_j = int(block_j)
                    new_i = i + block_i
                    new_j = j + block_j
                    if block_i == block_j == 0 or new_i < 0 or new_i > 8 or new_j < 0 or new_j > 8: #It needs to be modified according to the dimension of the light field image
                        continue
                    if likely[new_i, new_j] == 1:
                        G.add_edges(9 * new_i + new_j, 9 * i + j)
    return G


def samplePattern(numOfNodes, pattern):
# def samplePattern(numOfNodes):
    sampleMatrix = np.zeros((numOfNodes, numOfNodes))
    for i in range(int((numOfNodes - 1) / pattern)):
        sampleMatrix[pattern * (i + 1) - 1, pattern * (i + 1) - 1] = 1

    # A = [1, 7, 13, 15, 20, 25, 29, 33, 37, 43, 46, 49, 57, 59, 61, 71, 73, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
    #      89, 90, 91, 97, 98, 99, 109, 111, 113, 121, 124, 127, 133, 137, 141, 145, 150, 155, 157, 163, 169]
    # for i in A:
    #     sampleMatrix[i-1,i-1] = 1
    sampleMatrix[numOfNodes - 1, numOfNodes - 1] = 1
    return sampleMatrix


def readyuv(file_path, num_frm=169, height = 432, width = 624):
    Y_numpy = np.zeros((num_frm, height, width))
    fp = open(file_path, 'rb')
    for i in range(num_frm):
        for m in range(height):
            for n in range(width):
                Y_numpy[i, m, n] = ord(fp.read(1))
    fp.close()
    return Y_numpy

def readyuv_Z(file_path, num_frm=169, height = 432, width = 624):
    Y_numpy = np.zeros((num_frm, height, width))
    U_numpy = np.zeros((num_frm, height//2, width//2))
    V_numpy = np.zeros((num_frm, height//2, width//2))
    fp = open(file_path, 'rb')
    for i in range(num_frm):
        for m in range(height):
            for n in range(width):
                Y_numpy[i, m, n] = ord(fp.read(1))
        for m in range(height//2):
            for n in range(width//2):
                U_numpy[i, m, n] = ord(fp.read(1))
        for m in range(height//2):
            for n in range(width//2):
                V_numpy[i, m, n] = ord(fp.read(1))
    fp.close()
    return Y_numpy ,U_numpy ,V_numpy

def readyuv_ZE(file_path, num_frm=48, height = 432, width = 624):
    Y_numpy = np.zeros((num_frm, height, width))
    U_numpy = np.zeros((num_frm, height//2, width//2))
    V_numpy = np.zeros((num_frm, height//2, width//2))
    fp = open(file_path, 'rb')
    for i in range(num_frm):
        for m in range(height):
            for n in range(width):
                Y_numpy[i, m, n] = ord(fp.read(1))
        for m in range(height//2):
            for n in range(width//2):
                U_numpy[i, m, n] = ord(fp.read(1))
        for m in range(height//2):
            for n in range(width//2):
                V_numpy[i, m, n] = ord(fp.read(1))
    fp.close()
    return Y_numpy ,U_numpy ,V_numpy

def readyuv_edge(file_path, out_path, num_frm=112, height = 432, width = 624):
    Y_numpy = np.zeros((num_frm, height, width))
    U_numpy = np.zeros((num_frm, height//2, width//2))
    V_numpy = np.zeros((num_frm, height//2, width//2))
    fp = open(file_path, 'rb')
    for i in range(num_frm):
        for m in range(height):
            for n in range(width):
                Y_numpy[i, m, n] = ord(fp.read(1))
        for m in range(height//2):
            for n in range(width//2):
                U_numpy[i, m, n] = ord(fp.read(1))
        for m in range(height//2):
            for n in range(width//2):
                V_numpy[i, m, n] = ord(fp.read(1))
    fp.close()
    np.save(out_path, Y_numpy)
    return Y_numpy

def downsample(data, scale=2):
    num, height, width = data.shape
    output_ = np.zeros((num, height // scale, width // scale))
    output = np.zeros((num, height, width))
    for i in range(num):
        output_[i, :, :] = cv2.resize(data[i, :, :], (width // scale, height // scale), interpolation=cv2.INTER_CUBIC)
    for i in range(num):
        output[i, :, :] = cv2.resize(output_[i, :, :], (width, height), interpolation=cv2.INTER_CUBIC)
    output[output>255] = 255
    output[output<0] = 0
    return output


def gen_Ydata(file_path, scale=2, num_frm=169, name="bikes"):
    Y = readyuv(file_path)
    Y_label = np.zeros((798, num_frm, 32*32))
    Y_train = np.zeros((798, num_frm, 32*32))
    Y_t = downsample(Y, scale=scale)
    count = 0
    sample_matrix =  samplePattern(num_frm, 2)
    for x in range(0, 401, 20):
        for y in range(0, 593, 16):
            j=0
            for i in range(169):
                Y_label[count, j, :] = Y[i, x:x+32, y:y+32].reshape(1, 32*32)
                Y_train[count, j, :] = Y_t[i, x:x+32, y:y+32].reshape(1, 32*32)
                j+=1
            Y_train[count, :, :] = np.matmul(sample_matrix, Y_train[count, :, :])
            count+=1
    np.save('./testData/test_'+name, Y_train)
    np.save('./testData/label_'+name, Y_label)


def restore(data, num_frm = 169, height = 432, width = 624):
    Y_numpy = np.zeros((num_frm, height, width))
    count = 0
    for x in range(0, 401, 20):
        for y in range(0, 593, 16):
            for i in range(num_frm):
                Y_numpy[i, x:x+32, y:y+32] = data[count, i ,:].reshape(32, 32)
            count+=1
    return Y_numpy


def gen_Ydata_S(file_path, name = 'bikes'):
    Y = readyuv(file_path)
    Y_label = np.zeros((169, 432 * 624))
    Y_train = np.zeros((169, 432 * 624))
    Y_t = downsample(Y, scale=2)   #The resolution is 1/2 down sampling
    sample_matrix =  samplePattern(169, 2)
    for i in range(169):
        Y_label[i, :] = Y[i, :, :].reshape(1, 432 * 624)
        Y_train[i, :] = Y_t[i, :, :].reshape(1, 432 * 624)
    Y_train = np.matmul(sample_matrix, Y_train)
    np.save('./testData/S_test_'+name+'_low', Y_train)
    np.save('./testData/S_label_'+name+'_low', Y_label)

def gen_Ydata_Z(file_path, sample = 'oneHalf', name = 'bikes'):
    Y = readyuv_Z(file_path,num_frm=81,height=432,width=432)[0]
    Y_label = np.zeros((81, 432 * 432))
    Y_train = np.zeros((81, 432 * 432))
    # Y_t = downsample(Y, scale=2)   #The resolution is 1/2 down sampling
    Y_t = Y  # No sub-resolution sampling
    sample_matrix =  samplePattern(81,2)
    for i in range(81):
        Y_label[i, :] = Y[i, :, :].reshape(1, 432 * 432)
        Y_train[i, :] = Y_t[i, :, :].reshape(1, 432 * 432)
    Y_train = np.matmul(sample_matrix, Y_train)
    np.save('./testData/test/'+sample+'/S_test_'+name+'', Y_train)
    np.save('./testData/test/'+sample+'/S_label_'+name+'', Y_label)

def gen_Ydata_ZE(file_path, sample = 'oneHalf', name = 'bikes'):
    Y = readyuv_ZE(file_path,num_frm=41,height=512,width=512)[0]
    Y_label = np.zeros((81, 512 * 512))
    Y_train = np.zeros((81, 512 * 512))
    # Y_t = downsample(Y, scale=2)   #The resolution is 1/2 down sampling
    Y_t = Y  # No sub-resolution sampling
    # sample_matrix =  samplePattern(169, 2)
    sampleMatrix = np.zeros((169, 169))
    # A = [1, 7, 13, 15, 20, 25, 29, 33, 37, 43, 46, 49, 57, 59, 61, 71, 73, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
    #      89, 90, 91, 97, 98, 99, 109, 111, 113, 121, 124, 127, 133, 137, 141, 145, 150, 155, 157, 163, 169]
    # n = 0
    # for i in A:
    #     sampleMatrix[i-1,i-1] = 1
    # for i in A:
    #     Y_label[i-1, :] = Y[n, :, :].reshape(1, 432 * 624)
    #     Y_train[i-1, :] = Y_t[n, :, :].reshape(1, 432 * 624)
    #     n += 1
    #     print(n)
    for i in range(80):
        if i%2!=1:
            continue
        print(i)

        Y_label[i, :] = Y[i//2, :, :].reshape(1, 512 * 512)
        Y_train[i, :] = Y_t[i//2, :, :].reshape(1, 512 * 512)
    Y_label[80, :] = Y[40, :, :].reshape(1, 512 * 512)
    Y_train[80, :] = Y_t[40, :, :].reshape(1, 512 * 512)
    # Y_train = np.matmul(sample_matrix, Y_train)
    np.save('./testData/hevc/hci/S_test_'+name+'', Y_train)

def restore_S(data):
    Y = np.zeros((81, 432, 432))
    for i in range(81):
        Y[i, :, :] = data[i].reshape(432, 432)
    return Y


def creat_graph_sr_S(num_nodes=169):
    G = dgl.DGLGraph()
    G.add_nodes(num_nodes)
    numOfNodes = num_nodes
    pattern = 2
    size = int(numOfNodes ** 0.5)
    set = np.linspace(-pattern + 1, pattern - 1, num=2 * pattern - 1)
    sampleMatrix = samplePattern(numOfNodes, pattern)
    likely = np.diag(sampleMatrix).reshape(size, size)

    for i in range(size):
        for j in range(size):
            if likely[i, j] == 0:
                continue
            G.add_edges(13 * i + j, 13 * i + j)
            for block_i in set:
                block_i = int(block_i)
                for block_j in set:
                    block_j = int(block_j)
                    new_i = i + block_i
                    new_j = j + block_j
                    if block_i == block_j == 0 or new_i < 0 or new_i > 12 or new_j < 0 or new_j > 12:
                        continue
                    if likely[new_i, new_j] == 1:
                        G.add_edges(13 * new_i + new_j, 13 * i + j)
    return G

if __name__ == '__main__':
    # This part needs to be modified according to the specific situation
    # dataName = 'boxes'

    # Get raw data
    # Y = readyuv_Z('./lfData9/'+dataName+'_r.yuv',num_frm=81,height=432,width=432)[0]
    # np.save('./result/ori/9/ori_'+dataName+'', Y)

    # Get complete training data ,they are Y and label in train.py
    # gen_Ydata_Z('./lfData9/boxes_r.yuv','9','boxes')

    # Get the test data after codec
    # gen_Ydata_ZE('./hevcEncode/'+dataName+'_out.yuv','all',dataName,'10')
    # gen_Ydata_ZE('./testData/hevc/differentS/key/'+dataName+'_arstar_out.yuv','key',dataName)
    # gen_Ydata_ZE('./testData/hevc/hci/'+dataName+'_out.yuv','key',dataName)

    # Calculate test and label PSNR
    # bikes = np.load('./testData/S_test_bikes.npy')
    # bikes_l = np.load('./testData/S_label_bikes.npy')
    # bikes = restore_S(bikes)
    # bikes_l = restore_S(bikes_l)
    # for i in range(169):
    #     print(psnr(bikes[i, :, :], bikes_l[i, :, :]))

    # Calculate PSNR
    # dataName = 'bikes'
    # R = np.load('re_'+dataName+'_final.npy')
    # # Y_h = np.load('Y_h_'+dataName+'.npy')
    # Y = np.load('ori_'+dataName+'.npy')
    # all_psnr = 0
    # for i in range(169):
    #     # tmp = psnr(Y[i, :, :], Y_l[i, :, :] + Y_h[i, :, :])
    #     tmp = psnr(Y[i,:,:], R[i, :, :])
    #     all_psnr += tmp
    #     print(tmp)
    # print("the average of PSNR:%d ", all_psnr/169)

    # Calculate SSIM
    # Y_l = np.load('re_'+dataName+'.npy')
    # Y_h = np.load('Y_h_'+dataName+'.npy')
    # Y = np.load('ori_'+dataName+'.npy')
    # all_ssim = 0
    # for i in range(169):
    #     tmp = calculate_ssim(Y[i, :, :], Y_l[i, :, :] + Y_h[i, :, :])
    #     all_ssim += tmp
    #     # print(tmp)
    # print(all_ssim/169)

    # Generate PNG
    # dataName = 'bikes'
    # Y_l = np.load('ori_'+dataName+'.npy')
    # for i in range(169):
    #     cv2.imwrite('./bikesPng/'+str(i)+'.png',Y_l[i,:,:])