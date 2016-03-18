Tests for multiple GPU issue.

Had some trouble when using arrayfire with multiple GPUs. Arrayfire did not use the GPU specified with setDevice(). A workaround is to use the environment variable:
$ CUDA_VISIBLE_DEVICES=2,3 ./a.out
This will hide devices 0,1 from the application.

See also the corresponding thread in the af user group ("setDevice does not restrict arrayfire processes to this single device"):
https://groups.google.com/forum/#!topic/arrayfire-users/8G7cLKxEp7o
