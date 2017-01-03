local nn = require 'nn'
require 'cunn'

local function createModel(opt)

    local function gatedCNNLayer(inputFrameSize, outputFrameSize, kW)
        local gate = nn.Sequential()
        gate:add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, kW))
        gate:add(nn.Sigmoid())

        local res = nn.Sequential()
        res:add(nn.SpatialZeroPadding(0, 0, kW - 1, 0))
        res:add(nn.ConcatTable()
            :add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, kW))
            :add(gate))
        res:add(nn.CMulTable())

        return res
    end

    local function doubleBlock(frameSize, kW)
        local doubleLayer = nn.Sequential()
        doubleLayer:add(gatedCNNLayer(frameSize, frameSize, kW))
        doubleLayer:add(gatedCNNLayer(frameSize, frameSize, kW))

        local res = nn.Sequential()
        res:add(nn.ConcatTable()
            :add(doubleLayer)
            :add(nn.Identity()))
        res:add(nn.CAddTable(true))

        return res
    end

    local model = nn.Sequential()
    model:add(nn.LookupTable(opt.ivocabSize, opt.inputsize))
    model:add(gatedCNNLayer(opt.inputsize, opt.inputsize, opt.kW))
    model:add(nn.Dropout(0.5))
    for i = 1, opt.blockNum do
        model:add(doubleBlock(opt.inputsize, opt.kW))
        model:add(nn.Dropout(opt.dropout))
    end
    -- model:add(gatedCNNLayer(opt.inputsize, opt.inputsize, opt.kW))
    -- model:add(nn.Dropout(opt.dropout))
    model:add(nn.TemporalConvolution(opt.inputsize, opt.ivocabSize, 1))
    model:add(nn.View(-1, opt.ivocabSize))
    model:add(nn.LogSoftMax())

    return model
end

return createModel
