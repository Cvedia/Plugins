/*
	SPDX-FileCopyrightText: 2022 CVEDIA LTD

	SPDX-License-Identifier: Apache-2.0
*/
#include <algorithm>

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

int Post::labelToInt(cvec &labels, string label) {

	for (size_t idx=0;idx<labels.size();idx++)
	{
		auto key = *labels[idx]->getValue<string>();
		// convert key to lowercase
		std::transform(key.begin(), key.end(), key.begin(), ::tolower);

		if (key == label)
			return static_cast<int>(idx);
	}

	return -1;
}

expected<pCValue> Post::postYoloX(InferenceContext& ctx, std::vector<xt::xarray<float>> & output, CValue *inferenceConf) {
	pCValue ret = std::make_shared<CValue>();

	if (output.empty()) {
		return ret;
	}

	using namespace xt::placeholders;

	DetectorResults results;

	float confThreshold = 0.5F;
	bool filterEdgeDetections = false;
	bool nmsMergeBatches = false;
//	float nmsScoreThreshold = 0.1F;
	float nmsIouThreshold = 0.5F;
	bool hailoCompat = false;
	cvec labels;


	// Load defaults from model config
	auto modelConf = ctx.modelConfig;
	if (modelConf)
	{
		modelConf->getValueIfSet<bool>("filter_edge_detections", filterEdgeDetections);
		modelConf->getValueIfSet<bool>("nms_merge_batches", nmsMergeBatches);
		modelConf->getValueIfSet<float>("conf_threshold", confThreshold);
//		modelConf->getValueIfSet<float>("nms_score_threshold", nmsScoreThreshold);
		modelConf->getValueIfSet<float>("nms_iou_threshold", nmsIouThreshold);
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

	// Load runtime values from inference config
	inferenceConf->getValueIfSet<bool>("filter_edge_detections", filterEdgeDetections);
	inferenceConf->getValueIfSet<bool>("nms_merge_batches", nmsMergeBatches);
	inferenceConf->getValueIfSet<float>("conf_threshold", confThreshold);
//	inferenceConf->getValueIfSet<float>("nms_score_threshold", nmsScoreThreshold);
	inferenceConf->getValueIfSet<float>("nms_iou_threshold", nmsIouThreshold);

	auto remap = inferenceConf->getValueOr<cmap>("remap", {});

	IF_PLOG(plog::debug) {
		if (!api::state::isNodeSet("/flags/yolox_config")) {
			RT_CATCH(api::state::setNode("/flags/yolox_config", true));
			LOGD << "postYoloX.filter_edge_detections = " << filterEdgeDetections;
			LOGD << "postYoloX.nms_merge_batches = " << nmsMergeBatches;
			LOGD << "postYoloX.conf_threshold = " << confThreshold;
//			LOGD << "postYoloX.nms_score_threshold = " << nmsScoreThreshold;
			LOGD << "postYoloX.nms_iou_threshold = " << nmsIouThreshold;
			LOGD << "postYoloX: hailo compat? " << (hailoCompat ? "TRUE" : "FALSE");
		}
	}

	std::vector<int> classFilters;
	if (inferenceConf->hasChild("class_ids"))
	{
		cvec cv = inferenceConf->getValueOr("class_ids", cvec());
		for (auto const& c : cv) {
			int classid = c->getValue<int>().value();
			classFilters.push_back(classid);
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

	if (modelConf && modelConf->hasChild("labels")) {
		labels = modelConf->getChild("labels")->getValue<cvec>().value();
		//LOGD << "labels size: " << labels.size();

		if (hailoCompat) {
			if (labels.size() != shape[2] - 5) {
				LOGE << "Labels (" << labels.size() << ") does not match output shape (" << (shape[2]) << ")";
				return cvedia::rt::unexpected(RTErrc::InvalidArgument);
			}
		}
		else {
			if (labels.size() != shape[2] - 5) {
				LOGE << "Labels (" << labels.size() << ") does not match output shape (" << (shape[2] - 5) << ")";
				return cvedia::rt::unexpected(RTErrc::InvalidArgument);
			}
		}
	}

	auto dimPair = std::pair<int, int>(ctx.inputWidth, ctx.inputHeight);

	std::shared_lock<__shared_mutex_class> rlck(cacheMux_);
	size_t cacheCnt = cache_.count(dimPair);
	rlck.unlock();

	std::map<int, int> remapids;

	if (remap.size() > 0)
	{
		remapids.clear();

		for (auto const& kv : remap)
		{
			string key = kv.first;
			string label = kv.second->getValueOr<string>("");

			int keyid = labelToInt(labels, key);
			int replid = labelToInt(labels, label);
			if (keyid == -1)
			{
				LOGE << "Failed to lookup label " << key << " for remapping. Did the model change?";
				break;
			}
			if (replid == -1)
			{
				LOGE << "Failed to lookup label " << label << " for remapping. Did the model change?";
				break;
			}
			remapids[keyid] = replid;
		}
	}

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

	auto res = cvec();

	for (size_t batchIdx = 0; batchIdx < data.shape()[0]; batchIdx++) {
		auto layer = xt::view(data, batchIdx, xt::all());

		std::vector<size_t> dataShape(layer.shape().begin(), layer.shape().end());

		// Select the xywh coordinates 
		auto pred_x = xt::reshape_view(xt::view(layer, xt::all(), xt::keep(0)), { dataShape[0] });
		auto pred_y = xt::reshape_view(xt::view(layer, xt::all(), xt::keep(1)), { dataShape[0] });
		auto pred_w = xt::reshape_view(xt::view(layer, xt::all(), xt::keep(2)), { dataShape[0] });
		auto pred_h = xt::reshape_view(xt::view(layer, xt::all(), xt::keep(3)), { dataShape[0] });
		
		// Select the objectness confidence
		xt::xtensor<float, 1> predScores = xt::reshape_view(xt::view(layer, xt::all(), xt::range(4, 5)), { dataShape[0] });
		// Select confidence for all categories
		xt::xtensor<float, 2> predProb = xt::view(layer, xt::all(), xt::range(5, _));

		// Fetch index of highest confidence category
		xt::xtensor<int, 1> classMax = xt::argmax(predProb, -1);

		xt::xtensor<float, 1> classView = xt::amax(predProb, -1);

		// Multiply class conf by objectness score
		xt::noalias(predScores) = predScores * classView;

		// Filter by confidence threshold
		auto scoreMask = predScores > confThreshold;

		xt::xtensor<float, 1> xVec = xt::filter(pred_x, scoreMask);
		xt::xtensor<float, 1> yVec = xt::filter(pred_y, scoreMask);
		xt::xtensor<float, 1> wVec = xt::filter(pred_w, scoreMask);
		xt::xtensor<float, 1> hVec = xt::filter(pred_h, scoreMask);

		// Get scores using score mask
		auto scores = xt::filter(predScores, scoreMask);

		// Get classes using score mask
		auto classes = xt::filter(classMax, scoreMask);
		
		vector<Rect2f> bboxes;
		vector<float> scoresVec;
		vector<int> classids;

		size_t kCnt = scores.shape()[0];
		for (size_t k = 0; k < kCnt; k++) {
			float conf = scores(k);

			if (conf > confThreshold)
			{				
				int classId = classes(k);
				if (remapids.count(classId))
				{
					classId = remapids[classId];
				}

				if (classFilters.empty() || in_vector(classFilters, classId))
				{
					float x = xVec(k);
					float y = yVec(k);
					float w = wVec(k);
					float h = hVec(k);

					// Convert zero centered coords to x1/x2 format
					Rect2f cvRect(x - w * 0.5f, y - h * 0.5f, (w), (h));

					//LOGD << "\tpre nms box: " << x << "x " << y << "w " << w << "h " << h << " conf: " << conf << " cls: " << classId;

					bboxes.push_back(cvRect);
					scoresVec.push_back(conf);
					classids.push_back(classId);
				}
				//LOGD << "\tpre nms box: " << x << "x " << y << "w " << w << "h " << h << " conf: " << conf << " cls: " << classId;
			}
		}

		results.bbox.push_back(bboxes);
		results.confidence.push_back(scoresVec);
		results.classid.push_back(classids);
	}

//	RT_CATCH(api::util::detsToAbsCoords(results, ctx.rects, false, true, Size(ctx.inputWidth, ctx.inputHeight)));
	RT_CATCH(api::util::detsToAbsCoords(results, ctx.rects, false, true, Size(ctx.inputWidth, ctx.inputHeight)));
	if (filterEdgeDetections)
		api::util::filterEdgeDetections(results, ctx.rects, ctx.frameBuffer);
	results = api::util::NMS(results, nmsMergeBatches, confThreshold, nmsIouThreshold);

	for (size_t bidx = 0, bboxCnt = results.bbox.size(); bidx < bboxCnt; bidx++) {
		size_t maxidx = results.bbox[bidx].size();

		for (size_t idx = 0; idx < maxidx; idx++) {
			auto dt = cmap();

			if (ctx.cust.size() > bidx) {
				dt["custom"] = ctx.cust[bidx];
			}

			if (ctx.inferenceCount)
			{
				dt["id"] = VAL(std::to_string(static_cast<int>((*ctx.inferenceCount)++)));
			}

			dt["x"] = make_shared<CValue>(results.bbox[bidx][idx].x);
			dt["y"] = make_shared<CValue>(results.bbox[bidx][idx].y);
			dt["width"] = make_shared<CValue>(results.bbox[bidx][idx].width);
			dt["height"] = make_shared<CValue>(results.bbox[bidx][idx].height);
			dt["confidence"] = make_shared<CValue>(results.confidence[bidx][idx]);
			dt["classid"] = make_shared<CValue>(results.classid[bidx][idx]);

			int id = results.classid[bidx][idx];
			if (id == -1) {
				LOGE << "Negative label index";
			} else {
				if (labels.size() > static_cast<size_t>(id)) {
					dt["label"] = labels[static_cast<size_t>(id)];
				}
				else
				{
					dt["label"] = dt["classid"];
				}
			}

			res.push_back(make_shared<CValue>(dt));

			//LOGD << "\tpost nms box[" << idx << "]: " << results.bbox[bidx][idx].x << "x" << results.bbox[bidx][idx].y << "x" << results.bbox[bidx][idx].width << "x" << results.bbox[bidx][idx].height << " conf: " << results.confidence[bidx][idx] << " cls: " << results.classid[bidx][idx];
		}
	}

	//LOGD << "yoloX postpro completed";
	return make_shared<CValue>(res);
}

void registerHandler() {
	api::inference::registerPostprocessHandler("yolox", &Post::postYoloX);
}
