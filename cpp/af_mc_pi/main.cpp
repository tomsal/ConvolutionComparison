#include <stdio.h>
#include <arrayfire.h>

using namespace af;

void pi_function() {
  int n = 20e6; // 20 million random samples
  array x = randu(n,f32), y = randu(n,f32);
  // how many fell inside unit circle?
  float pi = 4.0 * sum<float>((sqrt(x*x + y*y)) < 1) / n;
}

int main() {
  printf("pi_function took %g seconds\n", timeit(pi_function));
  return 0;
}

