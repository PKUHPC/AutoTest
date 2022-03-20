#pragma once

#ifdef HICE_USE_EIGEN

#include "Eigen/Core"
#include "Eigen/Dense"

namespace hice {

// Common Eigen types that we will often use
template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenArrayMap =
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

using EigenOuterStride = Eigen::OuterStride<Eigen::Dynamic>;
using EigenInnerStride = Eigen::InnerStride<Eigen::Dynamic>;
using EigenStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using EigenOuterStridedMatrixMap = Eigen::
    Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenOuterStride>;
template <typename T>
using EigenOuterStridedArrayMap = Eigen::
    Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenOuterStride>;
template <typename T>
using ConstEigenOuterStridedMatrixMap = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
    0,
    EigenOuterStride>;
template <typename T>
using ConstEigenOuterStridedArrayMap = Eigen::Map<
    const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>,
    0,
    EigenOuterStride>;
template <typename T>
using EigenStridedMatrixMap = Eigen::
    Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenStride>;
template <typename T>
using EigenStridedArrayMap =
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenStride>;
template <typename T>
using ConstEigenStridedMatrixMap = Eigen::
    Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenStride>;
template <typename T>
using ConstEigenStridedArrayMap = Eigen::
    Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenStride>;

// 1-d array
template <typename T>
using EigenArrayXt = Eigen::Array<T, Eigen::Dynamic, 1>;
using EigenArrayXf = Eigen::ArrayXf;
using EigenArrayXd = Eigen::ArrayXd;
using EigenArrayXi = Eigen::ArrayXi;
using EigenArrayXb = EigenArrayXt<bool>;

// 2-d array, column major
template <typename T>
using EigenArrayXXt = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
using EigenArrayXXf = Eigen::ArrayXXf;

// 2-d array, row major
template <typename T>
using EigenRowArrayXXt =
    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenRowArrayXXf = EigenRowArrayXXt<float>;

// 1-d vector
template <typename T>
using EigenVectorXt = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using EigenVectorXd = Eigen::VectorXd;
using EigenVectorXf = Eigen::VectorXf;

// 1-d row vector
using EigenRowVectorXd = Eigen::RowVectorXd;
using EigenRowVectorXf = Eigen::RowVectorXf;

// 2-d matrix, column major
template <typename T>
using EigenMatrixXt = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
using EigenMatrixXd = Eigen::MatrixXd;
using EigenMatrixXf = Eigen::MatrixXf;

// 2-d matrix, row major
template <typename T>
using EigenRowMatrixXt =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenRowMatrixXd = EigenRowMatrixXt<double>;
using EigenRowMatrixXf = EigenRowMatrixXt<float>;

} // hice namespace

#endif