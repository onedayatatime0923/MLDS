
import torch
from torch.autograd import grad, Variable, backward
assert grad and Variable and backward

a=Variable(torch.cuda.FloatTensor([1,2,3]),requires_grad=True)
b=Variable(torch.cuda.FloatTensor([1,2,3]),requires_grad=True)
c=a.dot(b)

def Hessian(l,var):
    g=torch.cat(list(grad(c,var,retain_graph=True,create_graph= True)),0)
    h=[]
    for i in range(len(g)):
        e=torch.zeros(len(g)).cuda()
        e[i]=1
        e=Variable(e,requires_grad=True)
        h.append(torch.cat(list(grad(g.dot(e),var,retain_graph=True)),0).view(1,-1))
    h=torch.cat(h,0)
    print(h)
    return (h)
print(Hessian(c,[a,b]))


