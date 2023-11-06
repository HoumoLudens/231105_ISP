import torch
import torch.nn as nn
import numpy as np
import cv2

from torch_filter import torch_filter_auto


def start_auto(data, target):
    filter_auto=torch_filter_auto(3,is_grad=True)
    criterion = nn.MSELoss()
    optim = torch.optim.SGD(filter_auto.parameters(), lr = 1e-6) # 定义优化器

    for step in range(10000):
        predict = filter_auto(data) # 输出模型预测结果
        loss = criterion(predict, target) # 输出损失函数

        if step and step % 10 == 0:
            print("Loss:{:.8f}".format(loss.item()))

        optim.zero_grad() # 清零梯度
        loss.backward() # 反向传播
        optim.step()
    
def run(ori_img_file, target_img_file, gt_weight):
    ori_img=cv2.imread(ori_img_file)#输入图片路径
    ori_img = np.transpose((np.array(ori_img, np.float64))/255, [2, 0, 1])
    ori_img = torch.from_numpy(ori_img).type(torch.FloatTensor)
    ori_img = ori_img.unsqueeze(dim=0)
    
    target_img=cv2.imread(target_img_file)#输入图片路径
    target_img = np.transpose((np.array(target_img, np.float64))/255, [2, 0, 1])
    target_img = torch.from_numpy(target_img).type(torch.FloatTensor)
    target_img = target_img.unsqueeze(dim=0)
    
    start_auto(ori_img, target_img)
    
    return

def check_gpu():
    print(torch.__version__ )
    print(torch.cuda.is_available()) 
    print(torch.cuda.device(0))
    if torch.cuda.device_count() > 0:
        print(torch.cuda.get_device_name(0))

if __name__ == '__main__':
      
    ori_img = 'imgtest.jpg'
    target_img = 'new_img.jpg'
    
    gt_weight = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    
    check_gpu()
    # run(ori_img, target_img, gt_weight)
