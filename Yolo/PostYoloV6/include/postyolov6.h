/*
	SPDX-FileCopyrightText: 2022 CVEDIA LTD

	SPDX-License-Identifier: Apache-2.0
*/
#pragma once

#include "defines.h"

extern "C" EXPORT void registerHandler();
extern "C" EXPORT void logInit(plog::Severity severity, plog::IAppender * appender);

namespace cvedia {
	namespace rt {
		class Post
		{
		public:
			EXPORT static expected<pCValue> postYoloV6(internal::InferenceContext& ctx, std::vector<xt::xarray<float>>& output, CValue* inferenceConf);
		};
	}
}

