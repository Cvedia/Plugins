/*
	SPDX-FileCopyrightText: 2022 CVEDIA LTD

	SPDX-License-Identifier: Apache-2.0
*/
#include <xtensor/xadapt.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xpad.hpp>
#include <plog/Log.h>

#include "cvalue.h"
#include "api/inference.h"
#include "api/util.h"
#include "interface/inferencehandler.h"
#include "plog/Initializers/RollingFileInitializer.h"

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

extern "C" EXPORT void registerHandler();
extern "C" EXPORT void logInit(plog::Severity severity, plog::IAppender * appender);

void logInit(plog::Severity severity, plog::IAppender * appender)
{
	plog::init(severity, appender); // Initialize the shared library logger.
}

static expected<xt::xarray<float>> postYolov4_postprocess_bbbox(vector<xt::xarray<float>> pred_bbox, xt::xtensor<float, 3> ANCHORS, vector<float> STRIDES, vector<float> XYSCALE) {
	size_t size = pred_bbox.size();

	for (size_t i = 0; i < size; i++) {
		auto& pred = pred_bbox[i];

		auto conv_shape = pred.shape();
		auto output_size = conv_shape[1];

		// Retrieve xy & wh in the last dimension. XY is in center-form
		auto conv_raw_dxdy = xt::view(pred, xt::all(), xt::all(), xt::all(), xt::all(), xt::range(0, 2));
		auto conv_raw_dwdh = xt::view(pred, xt::all(), xt::all(), xt::all(), xt::all(), xt::range(2, 4));

		// Build a tensor with the size of the anchor arrangement
		auto xyGrid = xt::meshgrid(xt::arange(output_size), xt::arange(output_size));

		// Create a grid with 2 tensors
		auto gridStack = xt::stack(xt::xtuple(std::get<1>(xyGrid), std::get<0>(xyGrid)), 2);

		auto xy_grid2 = xt::expand_dims(gridStack, 2);
		auto xy_grid3 = xt::tile(xt::expand_dims(xy_grid2, 0), { 1, 1, 1, 3, 1 });

		// Perform expit() manually
		auto expit = (1. / (1 + xt::exp(-conv_raw_dxdy)));

		// Convert coordinates using anchor list
		auto predXy = ((expit * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1.f) + xy_grid3) * STRIDES[i];
		auto predWh = (xt::exp(conv_raw_dwdh) * xt::squeeze(xt::view(ANCHORS, xt::keep(i), xt::all(), xt::all())));

		// Create view on the xywh output
		auto predView = xt::view(pred, xt::all(), xt::all(), xt::all(), xt::all(), xt::range(0, 4));

		// Perform a manual concatenate (its weirdly slow otherwise)
		auto s = predXy.shape();
		auto predMerge = xt::xtensor<float, 5>({ s[0], s[1], s[2], s[3], 4 });

		// Merge them manually
		xt::view(predMerge, xt::all(), xt::all(), xt::all(), xt::all(), xt::range(0, 2)) = predXy;
		xt::view(predMerge, xt::all(), xt::all(), xt::all(), xt::all(), xt::range(2, 4)) = predWh;

		// Write data to view
		predView = predMerge;
	}

	vector<xt::xarray<float>> retVec;

	// Remove the anchor dimensions
	for (auto x : pred_bbox)
	{
		vector<int64_t> v(x.shape().begin(), x.shape().end());
		int h = static_cast<int>(x.shape()[x.shape().size() - 1]);

		auto t = x.reshape({ -1, h });
		retVec.push_back(t);
	}

	// Should possibly also be converted to a manual concatenate
	auto ret = xt::concatenate(xt::xtuple(retVec[0], retVec[1], retVec[2]), 0);

	return ret;
}

static expected<pCValue> postYolov4(InferenceContext& ctx, std::vector<xt::xarray<float>>& output, CValue* inferenceConf) {
	pCValue ret = std::make_shared<CValue>();

	using namespace xt::placeholders;

	DetectorResults results;

	float confThreshold = 0.5F;
	bool filterEdgeDetections = false;
	bool nmsMergeBatches = false;
	float nmsScoreThreshold = 0.1F;
	float nmsIouThreshold = 0.5F;
	cvec anch1;
	cvec cvStrides;
	cvec cvXyscale;
	cvec labels;
	vector<float> strides;
	vector<float> xyscale;

	// Load defaults from model config
	auto modelConf = ctx.modelConfig;
	modelConf->getValueIfSet<float>("nms_score_threshold", nmsScoreThreshold);
	modelConf->getValueIfSet<float>("nms_iou_threshold", nmsIouThreshold);
	modelConf->getValueIfSet<cvec>("anchors", anch1);
	modelConf->getValueIfSet<cvec>("strides", cvStrides);
	modelConf->getValueIfSet<cvec>("xyscale", cvXyscale);

	// Load runtime values from inference config
	inferenceConf->getValueIfSet<bool>("filter_edge_detections", filterEdgeDetections);
	inferenceConf->getValueIfSet<bool>("nms_merge_batches", nmsMergeBatches);
	inferenceConf->getValueIfSet<float>("conf_threshold", confThreshold);
	inferenceConf->getValueIfSet<float>("nms_score_threshold", nmsScoreThreshold);
	inferenceConf->getValueIfSet<float>("nms_iou_threshold", nmsIouThreshold);

	auto data = output[0];

	auto shape = data.shape();

	if (modelConf->hasChild("labels")) {
		labels = modelConf->getChild("labels")->getValue<cvec>().value();

		if (labels.size() != shape[4] - 5) {
			LOGE << "Labels (" << labels.size() << ") does not match output shape (" << (shape[4] - 5) << ")";
			return cvedia::rt::unexpected(RTErrc::InvalidArgument);
		}
	}

	// Convert CValue array to xtensor
	xt::xtensor<float, 3> anchors;

	for (size_t d1 = 0; d1 < anch1.size(); d1++) {
		auto anch2 = anch1[d1]->getValue<cvec>().value();

		for (size_t d2 = 0; d2 < anch2.size(); d2++) {
			auto anch3 = anch2[d2]->getValue<cvec>().value();

			for (size_t d3 = 0; d3 < anch3.size(); d3++) {
				auto x = anch3[d3]->getValue<float>().value();

				if (anchors.size() == 0) {
					anchors.resize({ anch1.size(), anch2.size(), anch3.size() });
				}

				anchors(d1, d2, d3) = x;
			}
		}
	}

	for (auto& cvStride : cvStrides)
	{
		auto x = cvStride->getValue<float>().value();
		strides.push_back(x);
	}

	for (auto& cvXy : cvXyscale)
	{
		auto x = cvXy->getValue<float>().value();
		xyscale.push_back(x);
	}

	// VALID output
	RT_TRY(auto outputBbox, postYolov4_postprocess_bbbox(output, anchors, strides, xyscale));

	auto& layer = outputBbox;

	// Select the xywh coordinates 
	auto predXywh = xt::view(layer, xt::all(), xt::range(0, 4));
	// Select the objectness confidence
	auto predConf = xt::view(layer, xt::all(), xt::range(4, 5));
	// Select confidence for all categories
	auto predProb = xt::view(layer, xt::all(), xt::range(5, _));

	// Fetch index of highest confidence category
	xt::xtensor<int, 1> classMax = xt::argmax(predProb, -1);

	// Convert confidences into index list for index_view
	// index_view cannot accept an xtensor so we need to manually build this
	xt::xtensor<int, 2> classIdx = xt::stack(xt::xtuple(xt::arange<int>(static_cast<int>(predXywh.shape()[0])), classMax), 1);
	vector<std::array<int, 2>> indices;

	for (size_t idx = 0; idx < classIdx.shape()[0]; idx++) {
		indices.push_back(std::array<int, 2>{classIdx(idx, 0), classIdx(idx, 1)});
	}

	// Select winning confidence for each anchor
	auto classView = xt::index_view(predProb, indices);
	auto classviewExp = xt::expand_dims(classView, static_cast<size_t>(1));

	// Multiply class conf by objectness score
	xt::xarray<float> predScores = (predConf * classviewExp);

	// Filter by confidence threshold
	auto score_mask = xt::squeeze(predScores) > confThreshold;

	// Couldn't get filter() to work with (N, 4) so manually splitting
	// filtering and merging
	auto predSplit = xt::split(predXywh, 4, 1);
	auto xVec = xt::filter(xt::squeeze(xt::xarray<float>(predSplit[0])), score_mask);
	vector<int64_t> v3(predSplit[0].shape().begin(), predSplit[0].shape().end());
	auto yVec = xt::filter(xt::squeeze(xt::xarray<float>(predSplit[1])), score_mask);
	auto wVec = xt::filter(xt::squeeze(xt::xarray<float>(predSplit[2])), score_mask);
	auto hVec = xt::filter(xt::squeeze(xt::xarray<float>(predSplit[3])), score_mask);

	// Get scores using score mask
	auto scores = xt::filter(xt::squeeze(predScores), score_mask);
	// Get classes using score mask
	auto classes = xt::filter(classMax, score_mask);
	
	vector<Rect2f> bboxes;
	vector<float> scoresVec;
	vector<int> classids;

	std::vector<int> classFilters;
	if(inferenceConf->hasChild("class_filters"))
	{
		cvec cv = inferenceConf->getValueOr("class_filters", cvec());
		for (auto c : cv) {
			int classid = c->getValue<int>().value();
			classFilters.push_back(classid);
		}		
	}
	
	size_t kCnt = scores.shape()[0];
	for (size_t k = 0; k < kCnt; k++) {

		float conf = scores(k);

		if (conf > confThreshold) {
			auto x = static_cast<float>(xVec(k));
			auto y = static_cast<float>(yVec(k));
			auto w = static_cast<float>(wVec(k));
			auto h = static_cast<float>(hVec(k));

			int classId = classes(k);

			if (classFilters.empty() || in_vector(classFilters, static_cast<int>(classId)))
			{

				// Convert zero centered coords to x1/x2 format
				Rect2f cvRect(static_cast<float>(x - (w * 0.5f)), static_cast<float>(y - (h * 0.5f)), w, h);

				//LOGD << "\tbox: " << x << "x " << y << "w " << w << "h " << h << " conf: " << conf << " cls: " << classId;

				bboxes.push_back(cvRect);
				scoresVec.push_back(conf);
				classids.push_back(classId);
			}

		}
	}

	results.bbox.push_back(bboxes);
	results.confidence.push_back(scoresVec);
	results.classid.push_back(classids);

	auto res = cvec();

	RT_CATCH(api::util::detsToAbsCoords(results, ctx.rects, false, true, Size(ctx.inputWidth, ctx.inputHeight)));
	if (filterEdgeDetections)
		api::util::filterEdgeDetections(results, ctx.rects, ctx.frameBuffer);
	results = api::util::NMS(results, nmsMergeBatches, nmsScoreThreshold, nmsIouThreshold);

	for (size_t bidx = 0, bboxCnt = results.bbox.size(); bidx < bboxCnt; bidx++) {
		size_t maxidx = results.bbox[bidx].size();

		for (size_t idx = 0; idx < maxidx; idx++) {
			auto dt = cmap();

			//			int id = (int)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

			//			dt["id"] = make_shared<CValue>(id);
			if (ctx.inferenceCount) {
				dt["id"] = VAL(std::to_string(static_cast<int>((*ctx.inferenceCount)++)));
			}
			dt["custom"] = ctx.cust[bidx];
			dt["x"] = make_shared<CValue>(results.bbox[bidx][idx].x);
			dt["y"] = make_shared<CValue>(results.bbox[bidx][idx].y);
			dt["width"] = make_shared<CValue>(results.bbox[bidx][idx].width);
			dt["height"] = make_shared<CValue>(results.bbox[bidx][idx].height);
			dt["confidence"] = make_shared<CValue>(results.confidence[bidx][idx]);
			dt["classid"] = make_shared<CValue>(results.classid[bidx][idx]);
			int id = results.classid[bidx][idx];
			if (labels.size() > static_cast<size_t>(id)) {
				dt["label"] = labels[static_cast<size_t>(id)];
			}
			else
			{
				dt["label"] = dt["classid"];
			}

			res.push_back(make_shared<CValue>(dt));
		}
	}

	return make_shared<CValue>(res);
}

void registerHandler() {
	api::inference::registerPostprocessHandler("yolov4", postYolov4);
}
