/*
	SPDX-FileCopyrightText: 2022 CVEDIA LTD

	SPDX-License-Identifier: Apache-2.0
*/
#include "../../common/functions.h"

#include <plog/Log.h>
#include <plog/Init.h>

#include <xtensor/xtensor.hpp>

#include "postyolov5.h"

using std::make_shared;
using std::shared_ptr;
using std::string;
using std::size_t;

using namespace cvedia::rt;

void logInit(plog::Severity severity, plog::IAppender* appender) {
	init(severity, appender); // Initialize the shared library logger.
}

expected<pCValue> Post::postYoloV5(InferenceContext& ctx, std::vector<xt::xarray<float>>& output, CValue* inferenceConf) {
	auto ret = std::make_shared<CValue>();

	if (output.empty())
	{
		return ret;
	}

	auto const shape = output[0].shape();

	if (output.size() == 1 && shape.size() == 3)
	{
		// Format as per https://github.com/ultralytics/yolov5
		// python3.7 export.py --include onnx --nms --simplify
		if (!api::state::isNodeSet("/flags/yolov5_info"))
		{
			RT_CATCH(api::state::setNode("/flags/yolov5_info", true));
			LOGD << "YoloV5: Single output layer with dim 3 found. Using anchorsToBboxes()";
		}

		// Most likely we're receiving concatenated anchor boxes
		return anchorsToBboxes(ctx, output[0], inferenceConf);
	}
	else {
		LOGE << "YoloV5: Encountered unsupported output format.";
	}

	return {};
}

void registerHandler() {
	api::inference::registerPostprocessHandler("yolov5", &Post::postYoloV5);
}
