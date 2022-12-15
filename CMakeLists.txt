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
    subdirs("Yolo/PostYoloV4")
endif()

if (WITH_POST_YOLOV5_MODULE)
    subdirs("Yolo/PostYoloV5")
endif()

if (WITH_POST_YOLOV6_MODULE)
    subdirs("Yolo/PostYoloV6")
endif()

if (WITH_POST_YOLOV7_MODULE)
    subdirs("Yolo/PostYoloV7")
endif()

if (WITH_POST_YOLOX_MODULE)
    subdirs("Yolo/PostYoloX")
endif ()
