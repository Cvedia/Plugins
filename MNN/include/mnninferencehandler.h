/*
	SPDX-FileCopyrightText: 2022 CVEDIA LTD

	SPDX-License-Identifier: Apache-2.0
*/
#include <memory>
#include <string>
#include "mnncore.h"

#include "interface/inferencehandler.h"

#include "defines.h"

namespace cvedia {
	namespace rt {
		namespace module {
			class MNNInferenceHandler final : private MNNCore, public iface::InferenceHandler {
			public:
				EXPORT MNNInferenceHandler(std::string const& moduleName);
				EXPORT ~MNNInferenceHandler() override;

				EXPORT static std::shared_ptr<iface::InferenceHandler> create(std::string const& moduleName);

				EXPORT expected<void> loadBackend() override;
				EXPORT expected<void> loadModel(std::string const& path) override;

				EXPORT expected<void> readConfig();

				EXPORT std::vector<std::pair<std::string, std::string>> getDeviceGuids() override;

				EXPORT expected<std::vector<xt::xarray<float>>> runInference(std::vector<Tensor>& input) override;

				EXPORT expected<void> setDevice(std::string const& guid);

				EXPORT std::string getDefaultScheme() const override { return "mnn"; }
				EXPORT std::string getDefaultExtension() const override { return ".mnn"; }

				EXPORT pCValue getCapabilities() override;
				EXPORT std::vector<int> getInputShape() override;
				EXPORT std::vector<int> getOutputShape() override;
			private:
				pCValue modelConfNode_;
			};
		}
	}
}