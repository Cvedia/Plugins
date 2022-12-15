#pragma once

#include <plog/Log.h>

#include "defines.h"

#include <xtensor/xtensor.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xadapt.hpp>
#include <rterror.h>

#include "cvalue.h"
#include "api/inference.h"
#include "api/state.h"
#include "api/util.h"
#include "interface/inferencehandler.h"

using std::unique_lock;
using std::shared_lock;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::size_t;
using std::map;
using std::vector;

using namespace cvedia::rt;
using namespace internal;
using namespace xt::placeholders;

inline int labelToInt(cvec const& labels, string const label) {

	for (size_t idx = 0; idx < labels.size(); idx++)
	{
		auto key = *labels[idx]->getValue<string>();
		// convert key to lowercase
		std::transform(key.begin(), key.end(), key.begin(), tolower);

		if (key == label)
			return static_cast<int>(idx);
	}

	return -1;
}

inline std::map<int, int> buildRemapLookup(cmap remap, cvec labels) {
	std::map<int, int> remapids;

	if (remap.size() > 0)
	{
		remapids.clear();

		for (auto const& kv : remap)
		{
			string const key = kv.first;
			string const label = kv.second->getValueOr<string>("");

			int keyid = labelToInt(labels, key);
			int const replid = labelToInt(labels, label);
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

	return remapids;
}

inline std::vector<int> buildClassFilters(CValue* inferenceConf) {
	std::vector<int> classFilters;
	cvec const cv = inferenceConf->getValueOr("class_ids", cvec());
	for (auto const& c : cv)
	{
		int classid = c->getValue<int>().value();
		classFilters.push_back(classid);
	}

	return classFilters;
}

/*
 * \brief Convert Yolo style anchors into bboxes. The anchors need to be in 
 *		  absolute coordinates.
 */
inline expected<pCValue> anchorsToBboxes(InferenceContext& ctx, xt::xtensor<float, 3> data, CValue* inferenceConf) {

	float confThreshold = 0.5F;
	bool filterEdgeDetections = false;
	bool nmsMergeBatches = false;
	float nmsIouThreshold = 0.5F;
	cvec labels;

	// Load defaults from model config
	auto modelConf = ctx.modelConfig;
	if (modelConf)
	{
		modelConf->getValueIfSet<bool>("filter_edge_detections", filterEdgeDetections);
		modelConf->getValueIfSet<bool>("nms_merge_batches", nmsMergeBatches);
		modelConf->getValueIfSet<float>("conf_threshold", confThreshold);
		modelConf->getValueIfSet<float>("nms_iou_threshold", nmsIouThreshold);
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
	inferenceConf->getValueIfSet<float>("nms_iou_threshold", nmsIouThreshold);

	auto remap = inferenceConf->getValueOr<cmap>("remap", {});

	IF_PLOG(plog::debug)
	{
		if (!api::state::isNodeSet("/flags/yolo_config"))
		{
			RT_CATCH(api::state::setNode("/flags/yolo_config", true));
			LOGD << "postYolo.filter_edge_detections = " << filterEdgeDetections;
			LOGD << "postYolo.nms_merge_batches = " << nmsMergeBatches;
			LOGD << "postYolo.conf_threshold = " << confThreshold;
			LOGD << "postYolo.nms_iou_threshold = " << nmsIouThreshold;
		}
	}

	// Fetch labels from model config
	labels = modelConf->getValueOr<cvec>("labels", {});

	if (modelConf->hasChild("labels"))
	{
		if (labels.size() != data.shape()[2] - 5)
		{
			LOGE << "Labels (" << labels.size() << ") does not match output shape (" << (data.shape()[2] - 5) << ")";
			return cvedia::rt::unexpected(RTErrc::InvalidArgument);
		}
	}

	auto classFilters = buildClassFilters(inferenceConf);
	auto remapids = buildRemapLookup(remap, labels);

	DetectorResults results;

	for (size_t batchIdx = 0; batchIdx < data.shape()[0]; batchIdx++)
	{
		auto layer = view(data, batchIdx, xt::all());

		std::vector<size_t> dataShape(layer.shape().begin(), layer.shape().end());

		// Select the xywh coordinates 
		auto pred_x = reshape_view(view(layer, xt::all(), xt::keep(0)), {dataShape[0]});
		auto pred_y = reshape_view(view(layer, xt::all(), xt::keep(1)), {dataShape[0]});
		auto pred_w = reshape_view(view(layer, xt::all(), xt::keep(2)), {dataShape[0]});
		auto pred_h = reshape_view(view(layer, xt::all(), xt::keep(3)), {dataShape[0]});

		// Select the objectness confidence
		xt::xtensor<float, 1> predScores = reshape_view(view(layer, xt::all(), xt::range(4, 5)), {dataShape[0]});
		// Select confidence for all categories
		xt::xtensor<float, 2> predProb = view(layer, xt::all(), xt::range(5, _));

		// Fetch index of highest confidence category
		xt::xtensor<int, 1> classMax = argmax(predProb, -1);

		xt::xtensor<float, 1> classView = amax(predProb, -1);

		// Multiply class conf by objectness score
		noalias(predScores) = predScores * classView;

		// Filter by confidence threshold
		auto scoreMask = predScores > confThreshold;

		xt::xtensor<float, 1> xVec = filter(pred_x, scoreMask);
		xt::xtensor<float, 1> yVec = filter(pred_y, scoreMask);
		xt::xtensor<float, 1> wVec = filter(pred_w, scoreMask);
		xt::xtensor<float, 1> hVec = filter(pred_h, scoreMask);

		// Get scores using score mask
		auto scores = filter(predScores, scoreMask);

		// Get classes using score mask
		auto classes = filter(classMax, scoreMask);

		vector<Rect2f> bboxes;
		vector<float> scoresVec;
		vector<int> classids;

		size_t kCnt = scores.shape()[0];
		for (size_t k = 0; k < kCnt; k++)
		{
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

	auto res = cvec();

	RT_CATCH(api::util::detsToAbsCoords(results, ctx.rects, false, true, Size(ctx.inputWidth, ctx.inputHeight)));
	if (filterEdgeDetections)
		api::util::filterEdgeDetections(results, ctx.rects, ctx.frameBuffer);
	results = api::util::NMS(results, nmsMergeBatches, confThreshold, nmsIouThreshold);

	for (size_t bidx = 0, bboxCnt = results.bbox.size(); bidx < bboxCnt; bidx++)
	{
		size_t maxidx = results.bbox[bidx].size();

		for (size_t idx = 0; idx < maxidx; idx++)
		{
			auto dt = cmap();

			if (ctx.cust.size() > bidx)
			{
				dt["custom"] = ctx.cust[bidx];
			}

			if (ctx.inferenceCount)
			{
				dt["id"] = VAL(std::to_string(static_cast<int>((*ctx.inferenceCount)++)));
			}

			dt["x"] = VAL(results.bbox[bidx][idx].x);
			dt["y"] = VAL(results.bbox[bidx][idx].y);
			dt["width"] = VAL(results.bbox[bidx][idx].width);
			dt["height"] = VAL(results.bbox[bidx][idx].height);
			dt["confidence"] = VAL(results.confidence[bidx][idx]);
			dt["classid"] = VAL(results.classid[bidx][idx]);

			int id = results.classid[bidx][idx];
			if (id == -1)
			{
				LOGE << "Negative label index";
			}
			else
			{
				if (labels.size() > static_cast<size_t>(id))
				{
					dt["label"] = labels[static_cast<size_t>(id)];
				}
				else
				{
					dt["label"] = dt["classid"];
				}
			}

			res.push_back(VAL(dt));

			//LOGD << "\tpost nms box[" << idx << "]: " << results.bbox[bidx][idx].x << "x" << results.bbox[bidx][idx].y << "x" << results.bbox[bidx][idx].width << "x" << results.bbox[bidx][idx].height << " conf: " << results.confidence[bidx][idx] << " cls: " << results.classid[bidx][idx];
		}
	}

	//LOGD << "yoloV7 postpro completed";
	return VAL(res);
}

inline pCValue nmsToBboxes(InferenceContext& ctx, xt::xtensor<float, 2> data, CValue* inferenceConf) {
	auto res = cvec();

	auto const classFilters = buildClassFilters(inferenceConf);

	// Fetch labels from model config
	auto const modelConf = ctx.modelConfig;

	auto const labels = modelConf->getValueOr<cvec>("labels", {});
	auto const remap = inferenceConf->getValueOr<cmap>("remap", {});

	auto remapids = buildRemapLookup(remap, labels);

	float prevx1 = 0;
	float prevy1 = 0;
	float prevx2 = 0;
	float prevy2 = 0;

	for (size_t bboxIdx = 0; bboxIdx < data.shape()[0]; bboxIdx++)
	{
		int const batchIdx = static_cast<int>(data(bboxIdx, 0));

		auto const x1 = data(bboxIdx, 1);
		auto const y1 = data(bboxIdx, 2);
		auto const x2 = data(bboxIdx, 3);
		auto const y2 = data(bboxIdx, 4);

		// TensorRT pads the NMS output to a fixed size. Last item is on repeat
		if (x1 == prevx1 && y1 == prevy1 && x2 == prevx2 && y2 == prevy2)
			break;

		prevx1 = x1;
		prevy1 = y1;
		prevx2 = x2;
		prevy2 = y2;

		int classid = static_cast<int>(data(bboxIdx, 5));

		if (remapids.count(classid))
		{
			classid = remapids[classid];
		}

		if (classFilters.empty() || in_vector(classFilters, classid))
		{
			auto conf = data(bboxIdx, 6);

			auto dt = cmap();

			if (ctx.cust.size() > batchIdx)
			{
				dt["custom"] = ctx.cust[batchIdx];
			}

			if (ctx.inferenceCount)
			{
				dt["id"] = VAL(std::to_string(static_cast<int>((*ctx.inferenceCount)++)));
			}

			auto cvRect = Rect2f(x1, y1, x2 - x1, y2 - y1);

			auto const& srcRect = ctx.rects[batchIdx];

			float const xRatio = srcRect.width / static_cast<float>(ctx.inputWidth);
			float const yRatio = srcRect.height / static_cast<float>(ctx.inputHeight);

			auto const rx1 = cvRect.x * xRatio + srcRect.x;
			auto const ry1 = cvRect.y * yRatio + srcRect.y;
			auto const rx2 = cvRect.x * xRatio + srcRect.x + cvRect.width * xRatio;
			auto const ry2 = cvRect.y * yRatio + srcRect.y + cvRect.height * yRatio;

			cvRect = Rect2f(rx1, ry1, rx2 - rx1, ry2 - ry1);


			dt["x"] = make_shared<CValue>(cvRect.x);
			dt["y"] = make_shared<CValue>(cvRect.y);
			dt["width"] = make_shared<CValue>(cvRect.width);
			dt["height"] = make_shared<CValue>(cvRect.height);
			dt["confidence"] = make_shared<CValue>(conf);
			dt["classid"] = make_shared<CValue>(classid);

			int const id = classid;
			if (id == -1)
			{
				LOGE << "Negative label index";
			}
			else
			{
				if (labels.size() > static_cast<size_t>(id))
				{
					dt["label"] = labels[static_cast<size_t>(id)];
				}
				else
				{
					dt["label"] = dt["classid"];
				}
			}

			res.push_back(make_shared<CValue>(dt));
		}
	}

	return VAL(res);
}
