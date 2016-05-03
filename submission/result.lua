require 'nngraph'
require('base')
ptb = require('data')
model = torch.load("bestmodel/model.net87.367553621956")

function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        layers=2--model.ds
        for d = 1, layers do
            model.start_s[d]:zero()
        end
    end
end



function run_test()
    reset_state(state_test)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)
    
    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    g_enable_dropout(model.rnns)
end

if gpu then
    g_init_gpu(arg)
end


function transfer_data(x)
    if gpu then
        return x:cuda()
    else
        return x
    end
end

state_train = {data=transfer_data(ptb.traindataset(20))}
state_valid =  {data=transfer_data(ptb.validdataset(20))}
state_test =  {data=transfer_data(ptb.testdataset(20))}
local states = {state_train, state_valid, state_test}

for _, state in pairs(states) do
    reset_state(state)
end

run_test()
