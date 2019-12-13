#include <torch/extension.h>
#include <iostream>


at::Tensor computeHistogram(at::Tensor const &t, unsigned int numBins = 256);
void matchHistogram(at::Tensor &featureMaps, at::Tensor &targetHistogram);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("computeHistogram", &computeHistogram, "ComputeHistogram");
  m.def("matchHistogram", &matchHistogram, "MatchHistogram");
}
