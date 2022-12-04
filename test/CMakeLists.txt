# Add unit test files
list(APPEND aitisa_test_srcs
        tensor_test.cpp
        conv_test.cpp
        matmul_test.cpp
        broadcast_test.cpp
        binary_test.cpp
        pooling_test.cpp
        softmax_test.cpp
        slice_test.cpp
        dot_test.cpp
        cast_test.cpp
        duplicate_test.cpp
        dropout_test.cpp
        sqrt_test.cpp
        squeeze_test.cpp
        reshape_test.cpp
        batch_norm_test.cpp
        sigmoid_test.cpp
        relu_test.cpp
        horizontal_flip_test.cpp
        vertical_flip_test.cpp
        image_transpose_test.cpp
        random_horizontal_flip_test.cpp
        random_vertical_flip_test.cpp
        rot90_test.cpp
        rgb_to_yuv_test.cpp
        yuv_to_rgb_test.cpp
        rgb_to_yiq_test.cpp
        yiq_to_rgb_test.cpp
        rgb_to_hsv_test.cpp
        hsv_to_rgb_test.cpp
        rgb_to_grayscale_test.cpp
        grayscale_to_rgb_test.cpp
        adjust_brightness_test.cpp
        adjust_gamma_test.cpp
        adjust_contrast_test.cpp
        adjust_saturation_test.cpp
        adjust_hue_test.cpp
        random_brightness_test.cpp
        random_contrast_test.cpp
        random_hue_test.cpp
        random_saturation_test.cpp
        center_crop_test.cpp
        random_crop_test.cpp
        crop_to_bounding_box_test.cpp
        pad_to_bounding_box_test.cpp
        resize_with_crop_or_pad_test.cpp
        image_normalize_test.cpp
        image_gradients_test.cpp
        gaussian_blur_test.cpp
        cross_entropy_loss.cpp
        ctc_loss_test.cpp
        abs_test.cpp
        l1_loss_test.cpp
        square_test.cpp
        mse_test.cpp
        )

# Build a exe for each test files
foreach (test_src ${aitisa_test_srcs})
    get_filename_component(test_name ${test_src} NAME_WE)
    aitisa_api_register_exe(${test_name}
            "${test_src}" "test" "aitisa_api::aitisa_api" "aitisa_api::gmock")
endforeach ()

# Build a exe including all test files 
set(AITISA_BUILD_TEST_MAIN ON)
if (${AITISA_BUILD_TEST_MAIN})
    add_executable(main main.cpp ${aitisa_test_srcs})
    target_link_libraries(main "aitisa_api::aitisa_api" "aitisa_api::gmock")
endif ()