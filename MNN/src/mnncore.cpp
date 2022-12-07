/*
	SPDX-FileCopyrightText: 2022 CVEDIA LTD

	SPDX-License-Identifier: Apache-2.0
*/
#include "mnncore.h"
#include <xtensor/xadapt.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xtensor.hpp>

#include "builtin/functions.h"
#include <filesystem.hpp>

#include <cstring>
#include <iostream>

using std::make_shared;
using std::vector;
using std::string;
using std::map;
using std::unique_lock;
using std::mutex;

using namespace cvedia::rt;
using namespace cvedia::rt::module;

namespace fs = ghc::filesystem;

std::string shapeToString(std::vector<int> shape) {

	// Create string from Shape
	auto shapeStr = std::accumulate(std::begin(shape), std::end(shape), string(),
		[](string& ss, int& s)
		{
			return ss.empty() ? std::to_string(s) : ss + "x" + std::to_string(s);
		});

	return shapeStr;
}

MNNCore::MNNCore()
{
	deviceInputTensor_ = nullptr;
	hostInputTensor_ = nullptr;
	inSize_ = 0;
	modelLoaded_ = false;
	session_ = nullptr;
}

expected<void> MNNCore::loadBackend() {

	if (backendLoaded_) {
		unloadBackend();
	}

	unique_lock<mutex> m(sessMux_);

	backendLoaded_ = true;
	return {};
};

pCValue MNNCore::getCapabilities() {
	return VAL();
}

std::vector<std::pair<std::string, std::string>> MNNCore::getDeviceGuids() {
	std::vector<std::pair<std::string, std::string>> out;

	out.push_back(std::make_pair(string("auto"), "Runs on best available device"));

	return out;
}

expected<void> MNNCore::setDevice(std::string const& device) {

	LOGD << "Setting device to " << device;

	if (!loadBackend()) {
		return unexpected(RTErrc::NoSuchDevice);
	}

	return {};
}

expected<void> MNNCore::loadModel(string const& path) {

	auto weights = readFile(path);

	return loadModel(path, weights);
}

expected<void> MNNCore::loadModel(string const& path, std::vector<unsigned char> const& weights) {

	unique_lock<mutex> m(sessMux_);

	auto ptr = MNN::Interpreter::createFromBuffer(weights.data(), weights.size());

	if (ptr) {
		network_ = std::shared_ptr<MNN::Interpreter>(ptr);

		MNN::ScheduleConfig config;
		config.type = MNN_FORWARD_AUTO;

		MNN::BackendConfig backendConfig;
		backendConfig.precision = MNN::BackendConfig::Precision_Normal;
		backendConfig.memory = MNN::BackendConfig::Memory_Normal;
		backendConfig.power = MNN::BackendConfig::Power_Normal;

		config.backendConfig = &backendConfig;

		LOGD << "Creating network session";
		session_ = network_->createSession(config);
		if (!session_) {
			LOGE << "createSession returned nullptr";
			return unexpected(RTErrc::OperationFailed);
		}

		LOGD << "Input tensors";
		auto inputs = network_->getSessionInputAll(session_);

		if (inputs.size() != 1) {
			LOGE << "Found " << inputs.size() << " input tensors. Only one supported currently";
			return unexpected(RTErrc::UnsupportedModel);
		}

		for (auto const& input : inputs) {
			for (auto const& s : input.second->shape()) {
				inputShape_.push_back(s);
			}

			LOGD << "- " << input.first << " (" << shapeToString(inputShape_) << ")";

			deviceInputTensor_ = input.second;

			// Only support 1 input
			break;
		}

		LOGD << "Output tensors";
		auto outputs = network_->getSessionOutputAll(session_);
		for (auto const& output : outputs) {
			std::vector<int> op;
			
			op = output.second->shape();

			size_t total = 1;
			for (auto const& s : output.second->shape()) {
				total *= static_cast<size_t>(s);
			}

			std::stringstream outShapeStr;
			std::copy(op.begin(), op.end(), std::ostream_iterator<int>(outShapeStr, " "));

			deviceOutputTensors_.push_back(output.second);

			LOGD << "- " << output.first << " (" << shapeToString(op) << ")";

			outputShape_.push_back(op);
			outSize_.push_back(total);
		}

		hostInputTensor_ = new MNN::Tensor(deviceInputTensor_, deviceInputTensor_->getDimensionType());
		for (auto t : deviceOutputTensors_) {
			hostOutputTensors_.push_back(new MNN::Tensor(t, t->getDimensionType()));
		}

		modelLoaded_ = true;

		network_->releaseModel();

		return {};
	}
	else {
		modelLoaded_ = false;

		LOGE << "MNN failed to load model at " << path;
		return unexpected(RTErrc::LoadModelFailed);
	}
}

expected<vector<xt::xarray<float>>> MNNCore::runInference(std::vector<cvedia::rt::Tensor>& input) {

	unique_lock<mutex> m(sessMux_);

	vector<xt::xarray<float>> output;
	if (input.empty())
		return output;

	auto data = input[0].move<float>();
	memcpy(hostInputTensor_->host<float>(), data.data(), data.size() * sizeof(float));

	// Copy input data to MNN
	deviceInputTensor_->copyFromHostTensor(hostInputTensor_);

	network_->runSession(session_);

	for (size_t i = 0; i < outputShape_.size(); i++) {
		deviceOutputTensors_[i]->copyToHostTensor(hostOutputTensors_[i]);

		float* data = hostOutputTensors_[i]->host<float>();

		std::vector<size_t> sizetShape(outputShape_[i].begin(), outputShape_[i].end());
		auto xarr = xt::adapt(data, outSize_[i], xt::no_ownership(), sizetShape);
		output.push_back(xarr);
	}

	return output;
}

void MNNCore::unloadBackend() {

	unique_lock<mutex> m(sessMux_);

	backendLoaded_ = false;
}

MNNCore::~MNNCore()
{
	if (session_) {
		network_->releaseSession(session_);
	}

	unloadBackend();
}