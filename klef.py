import torch
import numpy as np

# o = torch.tensor(0.33)
# print(o)
# an = torch.sigmoid(o)
# print(an)

# [tensor([[-2.5184]])]
# [tensor([[-2.5184]])]
# predictions_op = torch.Tensor([[-2.5184]])
# predictions_op = torch.sigmoid(predictions_op)
# # print(s)
# print(predictions_op)
# ans = np.vstack((predictions_op)).ravel()
# print(ans)

l = [-2.5184078]
l = torch.Tensor(l)
print(l)
an = torch.sigmoid(l)
print(an)
np_arr = an.cpu().detach().numpy()
print(np_arr)

