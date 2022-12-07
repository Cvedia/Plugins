/*
	SPDX-FileCopyrightText: 2022 CVEDIA LTD

	SPDX-License-Identifier: Apache-2.0
*/
#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <mutex>
#include <memory>

#include <xtensor/xarray.hpp>
#include "builtin/tensor.h"

#include "builtin/tensor.h"

#include <MNN/MNNDefine.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <locale>
#include <codecvt>

#include "cvalue.h"

#include <plog/Log.h>

namespace cvedia {
	namespace rt {
		namespace module {
			/**
			 * \ingroup Core
			 * @{
			 */
			class MNNCore
			{
			public:
				struct config {
					int device_id = 0;
				};

				struct model_config {
					std::string channel_layout = "";
					float mean = 0;
					float stddev = 0;
					bool normalize_input = false;
					bool transpose_input = false;
				};

				struct stats {
				};

				MNNCore();
				~MNNCore();

				bool backendLoaded_ = false;

				expected<void> loadModel(std::string const& model);
				expected<void> loadModel(std::string const& model, std::vector<unsigned char> const& weights);
				expected<void> loadBackend();
				void unloadBackend();

				pCValue getCapabilities();

				expected<void> setDevice(std::string const& device);
				std::vector<std::pair<std::string, std::string>> getDeviceGuids();

				std::vector<int> getInputShape() {
					return inputShape_;
				}

				std::vector<int> getOutputShape() {
					return outputShape_[0];
				}

				expected<std::vector<xt::xarray<float>>> runInference(std::vector<cvedia::rt::Tensor>& input);

				config pluginConf;
				stats pluginStats;
				model_config modelConf;

				pCValue modelNode_;
			private:

				std::shared_ptr<MNN::Interpreter> network_;
				MNN::Session* session_;

				MNN::Tensor* deviceInputTensor_;
				std::vector<MNN::Tensor*> deviceOutputTensors_;

				MNN::Tensor* hostInputTensor_;
				std::vector<MNN::Tensor*> hostOutputTensors_;

				std::size_t inSize_;
				std::vector<std::size_t> outSize_;

				std::vector<int> inputShape_;
				std::vector<std::vector<int>> outputShape_;

				bool modelLoaded_;

				std::mutex sessMux_;

				std::vector<float> inputBatchTensor_;

				std::string curChannelLayout;
			};
		}
		/** @} */
	}
}
