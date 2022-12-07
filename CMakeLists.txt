# SPDX-FileCopyrightText: 2022 CVEDIA LTD
#
# SPDX-License-Identifier: Apache-2.0

if(WITH_MNN_MODULE)
    subdirs("MNN")
endif()

if (WITH_POST_CLASSIFIER_MODULE)
    subdirs("PostClassifier")
endif()

if (WITH_POST_YOLOV4_MODULE)
    subdirs("PostYoloV4")
endif()

if (WITH_POST_YOLOX_MODULE)
    subdirs("PostYoloX")
endif ()
