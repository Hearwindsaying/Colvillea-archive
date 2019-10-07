#pragma once

#ifndef COLVILLEA_DEVICE_TOOLKIT_MCSAMPLING_H_
#define COLVILLEA_DEVICE_TOOLKIT_MCSAMPLING_H_
#include <vector>
#include <memory>

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix.h>

#include "Utility.h"

namespace TwUtil
{
	namespace MonteCarlo
	{
		struct Distribution1D
		{
		public:
			Distribution1D(const float * f, int n) : func(f, f+n), cdf(n + 1)
			{
				const auto &N = n;
				cdf[0] = 0.f;
				for (auto i = 1; i < N + 1; ++i)
					cdf[i] = cdf[i - 1] + func[i - 1] / N;

				funcIntegral = cdf[N];
				if (funcIntegral == 0)
				{
					for (auto i = 1; i < N + 1; ++i)
						cdf[i] = static_cast<float>(i) / static_cast<float>(N);
				}
				else
				{
					for (auto i = 1; i < N + 1; ++i)
						cdf[i] /= funcIntegral;
				}
			}
		public:
			std::vector<float> func;
			std::vector<float> cdf;
			float funcIntegral;
		};

		struct Distribution2D
		{
		public:
			Distribution2D(const float * func, int nu, int nv)
			{
				this->pCondictionalV.reserve(nv);
				for (auto v = 0; v < nv; ++v)
					this->pCondictionalV.emplace_back(new Distribution1D(&func[v * nu], nu));
				//compute marginal sampling distribution p[v]
				std::vector<float> marginalFunc;
				marginalFunc.reserve(nv);
				for (auto v = 0; v < nv; ++v)
					marginalFunc.push_back(this->pCondictionalV[v]->funcIntegral);
				this->pMarginal.reset(new Distribution1D(&marginalFunc[0], nv));
			}
		public:
			std::vector<std::unique_ptr<Distribution1D>> pCondictionalV;//p(u|v)
			std::unique_ptr<Distribution1D> pMarginal;//p(v)
		};
	}
}

#endif // COLVILLEA_DEVICE_TOOLKIT_MCSAMPLING_H_