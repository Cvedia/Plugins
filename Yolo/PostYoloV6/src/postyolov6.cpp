/*
	SPDX-FileCopyrightText: 2022 CVEDIA LTD

	SPDX-License-Identifier: Apache-2.0
*/
#include "../../common/functions.h"

#include <plog/Log.h>
#include <plog/Init.h>

#include <xtensor/xtensor.hpp>

#include "postyolov6.h"

using std::make_shared;
using std::shared_ptr;
using std::string;
using std::size_t;

using namespace cvedia::rt;

void logInit(plog::Severity severity, plog::IAppender* appender) {
	init(severity, appender); // Initialize the shared library logger.
}

expected<pCValue> Post::postYoloV6(InferenceContext& ctx, std::vector<xt::xarray<float>>& output, CValue* inferenceConf) {
	auto ret = std::make_shared<CValue>();

	if (output.empty())
	{
		return ret;
	}

	auto const shape = output[0].shape();

	// Most likely we're receiving concatenated anchor boxes
	if (output.size() == 1 && shape.size() == 3)
	{
		// Format as per https://github.com/meituan/YOLOv6
		// ython3 ./deploy/ONNX/export_onnx.py --weights yolov6n.pt --img 640 --batch 1 --simplify
		if (!api::state::isNodeSet("/flags/yolov6_info"))
		{
			RT_CATCH(api::state::setNode("/flags/yolov6_info", true));
			LOGD << "YoloV6: Single output layer with dim 3 found. Using anchorsToBboxes()";
		}

		// Most likely we're receiving concatenated anchor boxes
		return anchorsToBboxes(ctx, output[0], inferenceConf);
	}
	else {
		LOGE << "YoloV6: Encountered unsupported output format.";
	}

	return {};
}

void registerHandler() {
	api::inference::registerPostprocessHandler("yolov6", &Post::postYoloV6);
}
