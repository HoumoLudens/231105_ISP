import torch
import numpy
import cv2
import numpy as np
import os

dir = os.curdir
from torch_filter import torch_filter
weight = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
net=torch_filter(weight,is_grad=False)
img=cv2.imread(r"imgtest.jpg")#输入图片路径
image = np.transpose((np.array(img, np.float64))/255, [2, 0, 1])
image = torch.from_numpy(image).type(torch.FloatTensor)
image = image.unsqueeze(dim=0)
image_sharp=net(image)
image_sharp=image_sharp.cpu().detach().numpy().copy().squeeze()
predictimag=np.transpose(image_sharp, [1, 2, 0])*255
filepath = dir + '/new_img.jpg'
cv2.imwrite(filepath,predictimag)#输出图片路径