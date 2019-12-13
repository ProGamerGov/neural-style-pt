#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREAD_COUNT 1024

__global__ void computeHistogram(float *tensor, float *histogram, float *minv, float *maxv, unsigned int channels, unsigned int tensorSize, unsigned int nBins)
{
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < channels * tensorSize)
    {
      // Compute which channel we're in
      unsigned int channel = index / tensorSize;
      // Normalize the value in range [0, numBins]
      float value = (tensor[index] - minv[channel]) / (maxv[channel] - minv[channel]) * float(nBins);
      // Compute bin index
      int bin = min((unsigned int)(value), nBins - 1);
      // Increment relevant bin
      atomicAdd(histogram + (channel * nBins) + bin, 1);
    }
}

// return cummulative histogram shifed to the right by 1
// ==> histogram[c][0] alweays == 0
__global__ void accumulateHistogram(float *histogram, unsigned int nBins)
{
  float t = 0;
  for (unsigned int i=0 ; i < nBins ; ++i)
    {
      float swap = histogram[i + blockIdx.x * nBins];
      histogram[i + blockIdx.x * nBins ] = t;
      t += swap;
    }
}

__global__ void buildSortedLinkmap(float *tensor, unsigned int *linkMap, float *cumulativeHistogram, unsigned int *localIndexes, long *indirection, float *minv, float *maxv, unsigned int channels, unsigned int tensorSize, unsigned int nBins)
{
  unsigned int index = threadIdx.x + blockIdx.x* blockDim.x;
  if (index < channels * tensorSize)
    {
      // Shuffle image -- Avoid the blurry top bug
      index = indirection[index];
      // Compute which channel we're in
      unsigned int channel = index / tensorSize;
      // Normalize the value in range [0, numBins]
      float value = (tensor[index] - minv[channel]) / (maxv[channel] - minv[channel]) * float(nBins);
      // Compute bin index
      int binIndex = min((unsigned int)(value), nBins - 1);
      // Increment and retrieve the number of pixel in said bin
      int localIndex = atomicAdd(&localIndexes[(channel * 256) + binIndex], 1);
      // Retrieve the number of pixel in all bin lower (in cummulative histogram)
      unsigned int lowerPixelCount = cumulativeHistogram[(channel * 256) + binIndex];
      // Set the linkmap for indes to it's position as "pseudo-sorted"
      linkMap[index] = lowerPixelCount + localIndex;
    }
}

__global__ void rebuild(float *tensor, unsigned int *linkMap, float *targetHistogram, float scale, unsigned int channels, unsigned int tensorSize)
{
  unsigned int index = threadIdx.x + blockIdx.x* blockDim.x;
  if (index < channels * tensorSize)
    {
      unsigned int channel = index / tensorSize;
      unsigned int value = 0;
      for (int i=0 ; i < 256 ; ++i)
	if (linkMap[index] >= targetHistogram[(channel * 256) + i] * scale) value = i;
      tensor[index] = (float)value;
    }
}

at::Tensor computeHistogram(at::Tensor const &t, unsigned int numBins)
{
  at::Tensor unsqueezed(t);
  unsqueezed = unsqueezed.cuda();

  if (unsqueezed.ndimension() == 1)
    unsqueezed.unsqueeze_(0);
  if (unsqueezed.ndimension() > 2)
    unsqueezed = unsqueezed.view({unsqueezed.size(0), -1});
  
  unsigned int c = unsqueezed.size(0);     // Number od channels
  unsigned int n = unsqueezed.numel() / c; // Number of element per channel
  at::Tensor min = torch::min_values(unsqueezed, 1, true).cuda();
  at::Tensor max = torch::max_values(unsqueezed, 1, true).cuda();

  at::Tensor h = at::zeros({int(c), int(numBins)}, unsqueezed.type()).cuda();
  computeHistogram<<<(c*n) / THREAD_COUNT + 1, THREAD_COUNT>>>(unsqueezed.data<float>(),
    							       h.data<float>(),
    							       min.data<float>(),
    							       max.data<float>(),
    							       c, n, numBins);

  return h;
}

void matchHistogram(at::Tensor &featureMaps, at::Tensor &targetHistogram)
{
  static std::map<unsigned int, at::Tensor> randomIndices;
  if (randomIndices[featureMaps.numel()].numel() != featureMaps.numel())
    randomIndices[featureMaps.numel()] = torch::randperm(featureMaps.numel()).to(at::kLong).cuda();

  at::Tensor unsqueezed(featureMaps);
  if (unsqueezed.ndimension() == 1)
    unsqueezed.unsqueeze_(0);
  if (unsqueezed.ndimension() > 2)
    unsqueezed = unsqueezed.view({unsqueezed.size(0), -1});

  unsigned int nBins = targetHistogram.size(1);
  unsigned int c = unsqueezed.size(0);     // Number of channels
  unsigned int n = unsqueezed.numel() / c; // Number of element per channel

  // Scale = numberOf Element in features / number of element in target
  float scale = float(featureMaps.numel()) / targetHistogram.sum().item<float>();
 
  at::Tensor featuresHistogram = computeHistogram(unsqueezed, nBins);
  accumulateHistogram<<<c, 1>>>(featuresHistogram.data<float>(), nBins);
  accumulateHistogram<<<c, 1>>>(targetHistogram.data<float>(), nBins);

  unsigned int *linkMap = NULL;
  cudaMalloc(&linkMap, c * n * sizeof(unsigned int));

  unsigned int *localIndexes = NULL;
  cudaMalloc(&localIndexes, c * nBins * sizeof(unsigned int));
  cudaMemset(localIndexes, 0, c * nBins * sizeof(unsigned int));

  at::Tensor min = torch::min_values(unsqueezed, 1, true).cuda();
  at::Tensor max = torch::max_values(unsqueezed, 1, true).cuda();

  buildSortedLinkmap<<<(c*n) / THREAD_COUNT + 1, THREAD_COUNT>>>(featureMaps.data<float>(), linkMap, featuresHistogram.data<float>(), localIndexes, randomIndices[featureMaps.numel()].data<long>(), min.data<float>(), max.data<float>(), c, n, nBins);
  rebuild<<<(c*n) / THREAD_COUNT + 1, THREAD_COUNT>>>(featureMaps.data<float>(), linkMap, targetHistogram.data<float>(), scale, c, n);

  featureMaps.div_(float(nBins));
  
  cudaFree(linkMap);
  cudaFree(localIndexes);
}
