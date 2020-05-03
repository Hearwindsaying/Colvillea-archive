#pragma once

#include <random>
#include <cmath>

/* _Sampling Fibonacci Directions_
 */
template<class Vector, size_t N>
inline __device__ void SamplingFibonacci(int nb, Vector *outvec) {

   // Golden ratio
   const float gratio = (sqrt(5.0f)+1.0f)/2.0f;

   for(int i=0; i<nb; ++i) {
      const float zi = 1.0f - float(2*i+1) / float(nb);
      const float xi = float(i) / gratio;
      const float pi = 2.0f*M_PI * (xi - std::floor(xi));

      const float sint = sqrt(1.0 - zi*zi);

      outvec[i] = (Vector(sint*cos(pi), sint*sin(pi), zi));
   }
}
