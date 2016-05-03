require 'torch'
require 'nn'
require 'nngraph'

x = nn.Linear(4, 2)()
y = nn.Linear(5,2)()
h1 = nn.Square()(nn.Tanh()(x))
h2 = nn.Square()(nn.Sigmoid()(y))
h3 = nn.CMulTable()({h1,h2})
z = nn.Identity()()
h4 = nn.CAddTable()({h3,z})
m = nn.gModule({x,y,z}, {h4})
--graph.dot(m.fg, 'MLP','gModule')

function backprop_gmod(x,y,z)
    grad = torch.Tensor({1,1})
    update=m:updateOutput({x,y,z})
    print("output: \n")
    print(update)
    bprop = m:updateGradInput({x, y,z}, grad)
    print("bprop to x: ")
    print(bprop[1])
    print("bprop to y: \n")
    print(bprop[2])
    print("bprop to z: \n")
    print(bprop[3])
end

x = torch.Tensor({.1,.1,.1,.1})
y = torch.Tensor({.2,.2,.2,.2,.2})
z = torch.Tensor({.3,.3})
backprop_gmod(x,y,z)
