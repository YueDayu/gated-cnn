require 'paths'
local model = require 'gated_cnn'
local dl = require 'dataload'
require 'nn'
local optim = require 'optim'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
-- training
cmd:option('--startlr', 0.001, 'learning rate at t=0')
cmd:option('--weightDecay', 0.00001, 'weight decay')
cmd:option('--minlr', 0.00001, 'minimum learning rate')
cmd:option('--saturate', 50, 'epoch at which linear decayed LR will reach minlr')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxnormout', -1, 'max l2-norm of each layer\'s output neuron weights')
cmd:option('--cutoff', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:option('--maxepoch', 1000, 'maximum number of epochs to run')
cmd:option('--show', 1000, 'show')
-- gated cnn layer 
cmd:option('--inputsize', 256, 'gated cnn size')
cmd:option('--kW', 3, 'convolution kernal size')
cmd:option('--blockNum', 4, 'block number')
cmd:option('--seqlen', 5, 'sequence length : back-propagate through time (BPTT) for this many time-steps')
cmd:option('--dropout', 0.2, 'dropout rate in model')
-- data
cmd:option('--batchsize', 32, 'number of examples per batch')
cmd:option('--trainsize', -1, 'number of train examples seen between each epoch')
cmd:option('--validsize', -1, 'number of valid examples used for early stopping and cross-validation') 
cmd:option('--savepath', paths.concat(dl.SAVE_PATH, 'gated_cnn'), 'path to directory where experiment log (includes model) will be saved')
cmd:option('--id', '', 'id string of this experiment (used to name output file) (defaults to a unique id)')

cmd:text()
local opt = cmd:parse(arg or {})

require 'cunn'
cutorch.setDevice(opt.device)

local trainset, validset, testset = dl.loadPTB({opt.batchsize, 1, 1})
opt.ivocabSize = #trainset.ivocab
opt.id = opt.id == '' and ('ptb' .. ':' .. dl.uniqueid()) or opt.id

local lm = model(opt)
local crit = nn.ClassNLLCriterion()
local preprocess = nn.Transpose({1, 2})

print(lm)

lm:cuda()
crit:cuda()
preprocess:cuda()

local xplog = {}
xplog.opt = opt -- save all hyper-parameters and such
xplog.dataset = 'PennTreeBank'
xplog.vocab = trainset.vocab
-- will only serialize params
xplog.model = lm
xplog.criterion = crit
xplog.preprocess = preprocess
-- keep a log of NLL for each epoch
xplog.trainppl = {}
xplog.valppl = {}
-- will be used for early-stopping
xplog.minvalppl = 99999999
xplog.epoch = 0
local ntrial = 0
paths.mkdir(opt.savepath)

local epoch = 1
opt.lr = opt.startlr
opt.trainsize = opt.trainsize == -1 and trainset:size() or opt.trainsize
opt.validsize = opt.validsize == -1 and validset:size() or opt.validsize

local params, grad_params = lm:getParameters()

local function feval()
    return crit.output, grad_params
end

local optimState = {
    learningRate = opt.lr,
    weightDecay = opt.weightDecay
    -- learningRateDecay = 0.0,
    -- momentum = opt.momentum,
    -- nesterov = true,
    -- dampening = 0.0
}

while opt.maxepoch <= 0 or epoch <= opt.maxepoch do
    print("")
    print("Epoch #"..epoch.." :")
    
    -- 1. training
    
    local a = torch.Timer()
    lm:training()
    local h = 0
    local h1 = 0
    local showErr = 0
    local sumErr = 0
    for i, inputs, targets in trainset:subiter(opt.seqlen, opt.trainsize) do
        targets = preprocess:forward(targets:float():cuda()):clone()
        inputs = preprocess:forward(inputs:float():cuda()):clone()
        targets = targets:view(-1)
        
        -- forward
        local outputs = lm:forward(inputs)
        local err = crit:forward(outputs, targets)
        sumErr = sumErr + err
        showErr = showErr + err
        h1 = h1 + 1
        h = h + 1

        if i % opt.show == 0 then
            showErr = showErr / h1
            print('  ' .. epoch .. '-' .. i .. '  err: ' .. string.format("%.2f", showErr) .. '  ppl: ' .. string.format("%.2f", torch.exp(showErr)) .. '  avgppl: ' .. string.format("%.2f", torch.exp(sumErr / h)))
            showErr = 0
            h1 = 0
        end
        
        -- backward 
        local gradOutputs = crit:backward(outputs, targets)
        lm:zeroGradParameters()
        lm:backward(inputs, gradOutputs)
        
        optim.adam(feval, params, optimState)
    
        if i % 1000 == 0 then
            collectgarbage()
        end
   end
   
   -- learning rate decay
    -- optimState.learningRate = optimState.learningRate + (opt.minlr - opt.startlr) / opt.saturate
    -- optimState.learningRate = math.max(opt.minlr, optimState.learningRate)

    print("learning rate", optimState.learningRate)
    -- if opt.meanNorm then
        -- print("mean gradParam norm", opt.meanNorm)
    -- end

    if cutorch then cutorch.synchronize() end
    local speed = a:time().real/opt.trainsize
    print(string.format("Speed : %f sec/batch ", speed))
    
    local ppl = torch.exp(sumErr / h)
    print("Training PPL : "..ppl)
    
    xplog.trainppl[epoch] = ppl
    
    -- 2. cross-validation
    
    lm:evaluate()
    local sumErr = 0
    local h = 0
    for i, inputs, targets in validset:subiter(opt.seqlen, opt.validsize) do
        targets = preprocess:forward(targets:float():cuda()):clone()
        inputs = preprocess:forward(inputs:float():cuda()):clone()
        targets = targets:view(-1)
        
        -- forward
        local outputs = lm:forward(inputs)
        local err = crit:forward(outputs, targets)
        sumErr = sumErr + err
        h = h + 1
    end
    
    local ppl = torch.exp(sumErr/h)
    -- Perplexity = exp( sum ( NLL ) / #w)
    print("Validation PPL : "..ppl)
    
    xplog.valppl[epoch] = ppl
    
    -- early-stopping
    if ppl < xplog.minvalppl then
        -- save best version of model
        xplog.minvalppl = ppl
        xplog.epoch = epoch 
        local filename = paths.concat(opt.savepath, opt.id..'.t7')
        print("Found new minima. Saving to "..filename)
        torch.save(filename, xplog)
    end
    
    collectgarbage()
    epoch = epoch + 1
end
