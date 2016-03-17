This code runs convolutions on random images in order to check the memory limits of the common 2D convolution and separable convolution.
separable convolution:
By default, it is restricted to work only with kernels smaller than 31. Afterwards, 2D FFT convolution is supposed to be faster.
common 2D convolution:
Uses FFT for large kernels, triggered by some heuristics. This may result in unexpected out of memory encounters.

See also af user group ("Errors for 2D convolution for large images and kernels. (C++, python)"):
https://groups.google.com/forum/#!topic/arrayfire-users/06rZP8NSpJQ
