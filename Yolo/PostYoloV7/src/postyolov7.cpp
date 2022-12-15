/*
	SPDX-FileCopyrightText: 2022 CVEDIA LTD

	SPDX-License-Identifier: Apache-2.0
*/
#include "../../common/functions.h"

#include <plog/Log.h>
#include <plog/Init.h>

#include "postyolov7.h"

using std::make_shared;
using std::shared_ptr;
using std::string;
using std::size_t;

using namespace cvedia::rt;

void logInit(plog::Severity severity, plog::IAppender* appender) {
	init(severity, appender); // Initialize the shared library logger.
}

expected<pCValue> Post::postYoloV7(InferenceContext& ctx, std::vector<xt::xarray<float>>& output, CValue* inferenceConf) {
	auto ret = std::make_shared<CValue>();

	if (output.empty())
	{
		return ret;
	}

	auto shape = output[0].shape();

	// Model most likely contained NMS
	if (output.size() == 1 && shape.size() == 2 && shape[1] == 7)
	{
		if (!api::state::isNodeSet("/flags/yolov7_info"))
		{
			RT_CATCH(api::state::setNode("/flags/yolov7_info", true));
			LOGD << "YoloV7: Single output layer with dim 2 found. Using nmsToBboxes()";
		}

		return nmsToBboxes(ctx, output[0], inferenceConf);
	} else if (output.size() == 1 && shape.size() == 3)
	{
		if (!api::state::isNodeSet("/flags/yolov7_info"))
		{
			RT_CATCH(api::state::setNode("/flags/yolov7_info", true));
			LOGD << "YoloV7: Single output layer with dim 3 found. Using anchorsToBboxes()";
		}

		// Most likely we're receiving concatenated anchor boxes
		return anchorsToBboxes(ctx, output[0], inferenceConf);
	}
	else {
		LOGE << "YoloV7: Encountered unsupported output format.";
	}

	return {};
}

void registerHandler() {
	api::inference::registerPostprocessHandler("yolov7", &Post::postYoloV7);
}
