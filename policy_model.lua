require 'nn'


local function ConvReLU(dnn, inputPlanes, outputPlanes, filterSize, step)
  local pad = (filterSize - 1) / 2
  dnn:add(nn.SpatialConvolution(inputPlanes, outputPlanes, filterSize, filterSize, step, step, pad, pad))
  dnn:add(nn.ELU())
  -- dnn:add(nn.SpatialBatchNormalization(outputPlanes, 1e-3))
  -- dnn:add(nn.ReLU(true))
  return dnn
end



local function BuildModel(numFeatures, numFilters, numConvLayers)
  local dnn = nn.Sequential()
  ConvReLU(dnn, numFeatures, numFilters, 5, 1)

  for l = 2,numConvLayers do
    ConvReLU(dnn, numFilters, numFilters, 3, 1)
  end

  ConvReLU(dnn, numFilters, 1, 1, 1)
  dnn:add(nn.SoftMax())
  return dnn
end


return BuildModel(48, 256, 5)
