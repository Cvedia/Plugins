/*
	SPDX-FileCopyrightText: 2022 CVEDIA LTD

	SPDX-License-Identifier: Apache-2.0
*/
#include <plog/Log.h>
#include <plog/Initializers/RollingFileInitializer.h>

#include "mnninferencehandler.h"

#include "internal/threadmanager.h"
#include "builtin/functions.h"

#include "api/inference.h"

#include <filesystem.hpp>

#ifdef WITH_MODELFORGE
#include "internal/modelforge.h"
#endif

using std::shared_ptr;
using std::string;
using std::vector;


using namespace cvedia::rt;
using namespace cvedia::rt::internal;
using namespace cvedia::rt::module;

namespace fs = ghc::filesystem;

extern "C" EXPORT void logInit(plog::Severity severity, plog::IAppender * appender)
{
	plog::init(severity, appender); // Initialize the shared library logger.
}

extern "C" EXPORT void registerHandler() {
	api::inference::registerSchemeHandler("mnn", &MNNInferenceHandler::create);
	api::inference::registerExtHandler(".mnn", &MNNInferenceHandler::create);
}

std::shared_ptr<iface::InferenceHandler> MNNInferenceHandler::create(string const& moduleName) {
	return std::make_shared<MNNInferenceHandler>(moduleName);
}

MNNInferenceHandler::MNNInferenceHandler(string const& moduleName) : InferenceHandler(moduleName) {
}

MNNInferenceHandler::~MNNInferenceHandler() {
}

std::vector<std::pair<std::string, std::string>> MNNInferenceHandler::getDeviceGuids() {
	std::vector<std::pair<std::string, std::string>> out;

	auto devices = MNNCore::getDeviceGuids();
	for (auto const& device : devices) {
		auto namelc = string("MNN");
		std::transform(namelc.begin(), namelc.end(), namelc.begin(), [](unsigned char c) { return static_cast<unsigned char>(std::tolower(static_cast<int>(c))); });

		auto guid = string(namelc) + "." + device.first;

		out.push_back(std::make_pair(guid, device.second));
	}

	return out;
}

expected<void> MNNInferenceHandler::loadBackend() {
	readConfig();

	return MNNCore::loadBackend();
}

expected<void> MNNInferenceHandler::loadModel(string const& path) {
	readConfig();

	TRY(MNNCore::loadModel(path));

	if (modelNode_) {
		modelConfNode_->setValue(*modelNode_);
	}

	return {};
}

expected<std::vector<xt::xarray<float>>> MNNInferenceHandler::runInference(std::vector<Tensor>& input) {
	return MNNCore::runInference(input);
}

expected<void> MNNInferenceHandler::readConfig() {
	getConfigWithReadLock(config);

	modelConfNode_ = config->getChildOrCreate("model");

	config->getValueIfSet<int>("device_id", pluginConf.device_id);

	return {};
}


expected<void> MNNInferenceHandler::setDevice(std::string const& guid) {
	return MNNCore::setDevice(guid);
}

pCValue MNNInferenceHandler::getCapabilities() {
	return MNNCore::getCapabilities();
}

std::vector<int> MNNInferenceHandler::getInputShape() {
	return MNNCore::getInputShape();
}

std::vector<int> MNNInferenceHandler::getOutputShape() {
	return MNNCore::getOutputShape();
}
