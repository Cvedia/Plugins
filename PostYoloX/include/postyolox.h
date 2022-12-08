/*
	SPDX-FileCopyrightText: 2022 CVEDIA LTD

	SPDX-License-Identifier: Apache-2.0
*/
#pragma once

#include <mutex>
#include "defines.h"

extern "C" EXPORT void registerHandler();
extern "C" EXPORT void logInit(plog::Severity severity, plog::IAppender * appender);

namespace cvedia {
	namespace rt {
		class Post
		{
			struct cache
			{
				xt::xtensor<float, 3> gridsConcat;
				xt::xtensor<float, 3> expandedConcat;
			};

		public:
			EXPORT static expected<pCValue> postYoloX(internal::InferenceContext& ctx, std::vector<xt::xarray<float>> & output, CValue* inferenceConf);
		private:
			static int labelToInt(cvec& labels, std::string label);

			static __shared_mutex_class cacheMux_;
			static std::map<std::pair<int, int>, cache> cache_;
		};
	}
}

