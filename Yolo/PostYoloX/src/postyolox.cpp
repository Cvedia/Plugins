/*
	SPDX-FileCopyrightText: 2022 CVEDIA LTD

	SPDX-License-Identifier: Apache-2.0
*/
#include <algorithm>

#include "../../common/functions.h"

#include <xtensor/xtensor.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>
#include "xtensor/xnoalias.hpp"
#include <plog/Log.h>

#include "builtin/buffermgr.h"

#include "api/config.h"
#include "cvalue.h"
#include "api/inference.h"
#include "api/state.h"
#include "api/util.h"
#include "interface/inferencehandler.h"
#include "plog/Init.h"

#include "postyolox.h"

#include "xtensor/xadapt.hpp"

using std::unique_lock;
using std::shared_lock;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::size_t;
using std::map;
using std::vector;

using namespace cvedia::rt;
using namespace cvedia::rt::internal;

std::map<std::pair<int, int>, Post::cache> Post::cache_;
__shared_mutex_class Post::cacheMux_;

void logInit(plog::Severity severity, plog::IAppender * appender)
{
	plog::init(severity, appender); // Initialize the shared library logger.
}

expected<pCValue> Post::postYoloX(InferenceContext& ctx, std::vector<xt::xarray<float>> & output, CValue *inferenceConf) {
	pCValue ret = std::make_shared<CValue>();

	if (output.empty()) {
		return ret;
	}

	bool hailoCompat = false;

	// Load defaults from model config
	auto modelConf = ctx.modelConfig;
	if (modelConf)
	{
		modelConf->getValueIfSet<bool>("hailo_compat", hailoCompat);
	}
	else
	{
		if (!api::state::isNodeSet("/flags/no_model_config"))
		{
			RT_CATCH(api::state::setNode("/flags/no_model_config", true));
			LOGW << "No model configuration loaded. Is this correct?";
		}
	}

	IF_PLOG(plog::debug) {
		if (!api::state::isNodeSet("/flags/yolox_config")) {
			RT_CATCH(api::state::setNode("/flags/yolox_config", true));
			LOGD << "postYoloX: hailo compat? " << (hailoCompat ? "TRUE" : "FALSE");
		}
	}

	xt::xtensor<float, 3> data;

	if (!hailoCompat) {
		xt::noalias(data) = output[0];
	}
	else {
		vector<xt::xtensor<float, 3>> outs;
		
		for (size_t i = 0; i < output.size(); i++) {
			auto e = output[i];
			// LOGD << "e[" << i << "] shape: " << xt::adapt(e.shape());
			xt::xarray<float> g = e.reshape({
				static_cast<int>(e.shape()[0]),
				-1,
				// works for nhwc and nchw
				static_cast<int>(e.shape()[3])
//				static_cast<int>(std::min(e.shape()[1], e.shape()[3]))
			});
			outs.push_back(g);
		}
		
		// LOGD << "out0 shape: " << xt::adapt(outs[0].shape());
		// LOGD << "out1 shape: " << xt::adapt(outs[1].shape());
		// LOGD << "out2 shape: " << xt::adapt(outs[2].shape());
		xt::noalias(data) = xt::concatenate(xt::xtuple(outs[0], outs[1], outs[2]), 1);

	}

	auto shape = data.shape(); // 1, 8400, 7
	//LOGD << "data shape: " << xt::adapt(shape);

	auto dimPair = std::pair<int, int>(ctx.inputWidth, ctx.inputHeight);

	std::shared_lock<__shared_mutex_class> rlck(cacheMux_);
	size_t cacheCnt = cache_.count(dimPair);
	rlck.unlock();

	if (cacheCnt == 0) {
		// Build the class id lookup table

		xt::xtensor<float, 1> strides = { 8, 16, 32 };
		xt::xtensor<float, 1> stridesmul = { 0.125, 0.0625, 0.03125 };

		xt::xtensor<float, 1> hsizes = { static_cast<float>(ctx.inputHeight), static_cast<float>(ctx.inputHeight), static_cast<float>(ctx.inputHeight) };
		xt::xtensor<float, 1> wsizes = { static_cast<float>(ctx.inputWidth), static_cast<float>(ctx.inputWidth), static_cast<float>(ctx.inputWidth) };

		xt::noalias(hsizes) *= stridesmul;
		xt::noalias(wsizes) *= stridesmul;

		std::array<xt::xtensor<float, 3>, 3> grids;

		vector<xt::xtensor<float, 3>> expandedStrides;
		for (size_t idx = 0; idx < strides.shape()[0]; idx++) {

			auto yvxv = xt::meshgrid(xt::arange(hsizes[idx]), xt::arange(wsizes[idx]));
			auto yv = std::get<0>(yvxv);
			auto xv = std::get<1>(yvxv);

			auto grid = xt::stack(xt::xtuple(xv, yv), 2).reshape({ 1, -1, 2 });
			grids[idx] = grid;

			size_t shape0 = grid.shape()[0];
			size_t shape1 = grid.shape()[1];

			xt::xtensor<float, 3> xs = xt::zeros<float>({ shape0, shape1, static_cast<size_t>(1) });

			auto full = xt::full_like(xs, strides[idx]);

			expandedStrides.emplace_back(full);
		}

		std::unique_lock<__shared_mutex_class> lck(cacheMux_);
		cache_[dimPair].gridsConcat = xt::concatenate(xt::xtuple(grids[0], grids[1], grids[2]), 1);
		cache_[dimPair].expandedConcat = xt::concatenate(xt::xtuple(expandedStrides[0], expandedStrides[1], expandedStrides[2]), 1);
	}

	std::shared_lock<__shared_mutex_class> lck(cacheMux_);
	auto &gridsConcat = cache_[dimPair].gridsConcat;
	auto &expandedConcat = cache_[dimPair].expandedConcat;
	lck.unlock();

	auto outputXY = xt::view(data, xt::all(), xt::all(), xt::range(0, 2));
	auto outputWH = xt::view(data, xt::all(), xt::all(), xt::range(2, 4));

	// LOGD << "\toutputXY: " << xt::adapt(outputXY.shape()); // 1 140 2
	// LOGD << "\toutputWH: " << xt::adapt(outputWH.shape()); // 1 140 2
	// LOGD << "\tgridsConcat: " << xt::adapt(gridsConcat.shape()); // 1 8400 2
	// LOGD << "\texpandedConcat: " << xt::adapt(expandedConcat.shape()); // 1 8400 1

	if (outputXY.shape()[1] != gridsConcat.shape()[1])
	{
		LOGE << "Incompatibles shapes. Are the inputWidth and inputHeight correct?";
		return cvedia::rt::unexpected(RTErrc::InvalidArgument);
	}

	xt::noalias(outputXY) = (outputXY + gridsConcat) * expandedConcat;
	xt::noalias(outputWH) = xt::exp(outputWH) * expandedConcat;

	return anchorsToBboxes(ctx, data, inferenceConf);
}

void registerHandler() {
	api::inference::registerPostprocessHandler("yolox", &Post::postYoloX);
}
