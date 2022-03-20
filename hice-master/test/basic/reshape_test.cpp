#include "hice/core/tensor.h"
#include "hice/core/shape_util.h"
#include "hice/core/index_util.h"
#include "hice/core/tensor_printer.h"
#include "hice/basic/factories.h"
#include "hice/basic/reshape.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace hice {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(OutOfPlaceReshapeTest, WithDefaultLayout) {
  Tensor scalar = hice::full({}, 1, dtype(kFloat).device(kCPU));
  Tensor reshaped_scalar = hice::reshape(scalar, {});
  EXPECT_EQ(reshaped_scalar.ndim(), 0);
  EXPECT_EQ(reshaped_scalar.size(), scalar.size());
  EXPECT_THAT(reshaped_scalar.dims(), IsEmpty());
  EXPECT_THAT(reshaped_scalar.layout().minor_to_major(), IsEmpty());
  EXPECT_EQ(&scalar.impl().storage().impl(),
            &reshaped_scalar.impl().storage().impl());

  Tensor vector = hice::full({2}, 1, dtype(kFloat).device(kCPU));
  Tensor reshaped_vector = hice::reshape(vector, {1, 2});
  EXPECT_EQ(reshaped_vector.ndim(), 2);
  EXPECT_EQ(reshaped_vector.size(), vector.size());
  EXPECT_THAT(reshaped_vector.dims(), ElementsAre(1, 2));
  EXPECT_THAT(reshaped_vector.layout().minor_to_major(), ElementsAre(1, 0));
  EXPECT_EQ(&vector.impl().storage().impl(),
            &reshaped_vector.impl().storage().impl());

  Tensor matrix = hice::full({3, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor reshaped_matrix = hice::reshape(matrix, {3, 1, 4});
  EXPECT_EQ(reshaped_matrix.ndim(), 3);
  EXPECT_EQ(reshaped_matrix.size(), matrix.size());
  EXPECT_THAT(reshaped_matrix.dims(), ElementsAre(3, 1, 4));
  EXPECT_THAT(reshaped_matrix.layout().minor_to_major(), ElementsAre(2, 1, 0));
  EXPECT_EQ(&matrix.impl().storage().impl(),
            &reshaped_matrix.impl().storage().impl());

  Tensor matrix2 = hice::full({24, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor reshaped_matrix2 = hice::reshape(matrix2, {3, 8, 4});
  EXPECT_EQ(reshaped_matrix2.ndim(), 3);
  EXPECT_EQ(reshaped_matrix2.size(), matrix2.size());
  EXPECT_THAT(reshaped_matrix2.dims(), ElementsAre(3, 8, 4));
  EXPECT_THAT(reshaped_matrix2.layout().minor_to_major(), ElementsAre(2, 1, 0));
  EXPECT_EQ(&matrix2.impl().storage().impl(),
            &reshaped_matrix2.impl().storage().impl());

  Tensor tensor = hice::full({4, 7, 4, 6}, 1, dtype(kFloat).device(kCPU));
  Tensor reshaped_tensor = hice::reshape(tensor, {28, 24});
  EXPECT_EQ(reshaped_tensor.ndim(), 2);
  EXPECT_EQ(reshaped_tensor.size(), tensor.size());
  EXPECT_THAT(reshaped_tensor.dims(), ElementsAre(28, 24));
  EXPECT_THAT(reshaped_tensor.layout().minor_to_major(), ElementsAre(1, 0));
  EXPECT_EQ(&tensor.impl().storage().impl(),
            &reshaped_tensor.impl().storage().impl());

  Tensor tensor2 = hice::full({4, 7, 8, 6}, 1, dtype(kFloat).device(kCPU));
  Tensor reshaped_tensor2 = hice::reshape(tensor2, {32, 42});
  EXPECT_EQ(reshaped_tensor2.ndim(), 2);
  EXPECT_EQ(reshaped_tensor2.size(), tensor2.size());
  EXPECT_THAT(reshaped_tensor2.dims(), ElementsAre(32, 42));
  EXPECT_THAT(reshaped_tensor2.layout().minor_to_major(), ElementsAre(1, 0));
  EXPECT_EQ(&tensor2.impl().storage().impl(),
            &reshaped_tensor2.impl().storage().impl());
}

TEST(OutOfPlaceExpandDimsTest, WithDefaultLayout) {
  Tensor scalar = hice::full({}, 1, dtype(kFloat).device(kCPU));
  Tensor expanded_scalar = hice::expand_dims(scalar, 0);
  EXPECT_EQ(expanded_scalar.ndim(), scalar.ndim() + 1);
  EXPECT_EQ(expanded_scalar.size(), scalar.size());
  EXPECT_THAT(expanded_scalar.dims(), ElementsAre(1));
  EXPECT_THAT(expanded_scalar.layout().minor_to_major(), ElementsAre(0));
  EXPECT_EQ(&scalar.impl().storage().impl(),
            &expanded_scalar.impl().storage().impl());

  Tensor scalar2 = hice::full({}, 1, dtype(kFloat).device(kCPU));
  Tensor expanded_scalar2 = hice::expand_dims(scalar2, -1);
  EXPECT_EQ(expanded_scalar2.ndim(), scalar2.ndim() + 1);
  EXPECT_EQ(expanded_scalar2.size(), scalar2.size());
  EXPECT_THAT(expanded_scalar2.dims(), ElementsAre(1));
  EXPECT_THAT(expanded_scalar2.layout().minor_to_major(), ElementsAre(0));
  EXPECT_EQ(&scalar2.impl().storage().impl(),
            &expanded_scalar2.impl().storage().impl());

  Tensor vector = hice::full({2}, 1, dtype(kFloat).device(kCPU));
  Tensor expanded_vector = hice::expand_dims(vector, 0);
  EXPECT_EQ(expanded_vector.ndim(), vector.ndim() + 1);
  EXPECT_EQ(expanded_vector.size(), vector.size());
  EXPECT_THAT(expanded_vector.dims(), ElementsAre(1, 2));
  EXPECT_THAT(expanded_vector.layout().minor_to_major(), ElementsAre(1, 0));
  EXPECT_EQ(&vector.impl().storage().impl(),
            &expanded_vector.impl().storage().impl());

  Tensor vector2 = hice::full({2}, 1, dtype(kFloat).device(kCPU));
  Tensor expanded_vector2 = hice::expand_dims(vector2, 1);
  EXPECT_EQ(expanded_vector2.ndim(), vector2.ndim() + 1);
  EXPECT_EQ(expanded_vector2.size(), vector2.size());
  EXPECT_THAT(expanded_vector2.dims(), ElementsAre(2, 1));
  EXPECT_THAT(expanded_vector2.layout().minor_to_major(), ElementsAre(1, 0));
  EXPECT_EQ(&vector2.impl().storage().impl(),
            &expanded_vector2.impl().storage().impl());

  Tensor matrix = hice::full({3, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor expanded_matrix = hice::expand_dims(matrix, 1);
  EXPECT_EQ(expanded_matrix.ndim(), matrix.ndim() + 1);
  EXPECT_EQ(expanded_matrix.size(), matrix.size());
  EXPECT_THAT(expanded_matrix.dims(), ElementsAre(3, 1, 4));
  EXPECT_THAT(expanded_matrix.layout().minor_to_major(), ElementsAre(2, 1, 0));
  EXPECT_EQ(&matrix.impl().storage().impl(),
            &expanded_matrix.impl().storage().impl());

  Tensor matrix2 = hice::full({3, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor expanded_matrix2 = hice::expand_dims(matrix2, -3);
  EXPECT_EQ(expanded_matrix2.ndim(), matrix.ndim() + 1);
  EXPECT_EQ(expanded_matrix2.size(), matrix.size());
  EXPECT_THAT(expanded_matrix2.dims(), ElementsAre(1, 3, 4));
  EXPECT_THAT(expanded_matrix2.layout().minor_to_major(), ElementsAre(2, 1, 0));
  EXPECT_EQ(&matrix2.impl().storage().impl(),
            &expanded_matrix2.impl().storage().impl());

  Tensor matrix3 = hice::full({3, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor expanded_matrix3 = hice::expand_dims(matrix3, -1);
  EXPECT_EQ(expanded_matrix3.ndim(), matrix.ndim() + 1);
  EXPECT_EQ(expanded_matrix3.size(), matrix.size());
  EXPECT_THAT(expanded_matrix3.dims(), ElementsAre(3, 4, 1));
  EXPECT_THAT(expanded_matrix3.layout().minor_to_major(), ElementsAre(2, 1, 0));
  EXPECT_EQ(&matrix3.impl().storage().impl(),
            &expanded_matrix3.impl().storage().impl());

  Tensor tensor = hice::full({3, 1, 4, 6}, 1, dtype(kFloat).device(kCPU));
  Tensor expanded_tensor = hice::expand_dims(tensor, 0);
  EXPECT_EQ(expanded_tensor.ndim(), tensor.ndim() + 1);
  EXPECT_EQ(expanded_tensor.size(), tensor.size());
  EXPECT_THAT(expanded_tensor.dims(), ElementsAre(1, 3, 1, 4, 6));
  EXPECT_THAT(expanded_tensor.layout().minor_to_major(),
              ElementsAre(4, 3, 2, 1, 0));
  EXPECT_EQ(&tensor.impl().storage().impl(),
            &expanded_tensor.impl().storage().impl());

  Tensor tensor2 = hice::full({3, 1, 4, 6}, 1, dtype(kFloat).device(kCPU));
  Tensor expanded_tensor2 = hice::expand_dims(tensor2, -1);
  EXPECT_EQ(expanded_tensor2.ndim(), tensor2.ndim() + 1);
  EXPECT_EQ(expanded_tensor2.size(), tensor2.size());
  EXPECT_THAT(expanded_tensor2.dims(), ElementsAre(3, 1, 4, 6, 1));
  EXPECT_THAT(expanded_tensor2.layout().minor_to_major(),
              ElementsAre(4, 3, 2, 1, 0));
  EXPECT_EQ(&tensor2.impl().storage().impl(),
            &expanded_tensor2.impl().storage().impl());

  Tensor tensor3 = hice::full({3, 1, 4, 6}, 1, dtype(kFloat).device(kCPU));
  Tensor expanded_tensor3 = hice::expand_dims(tensor3, -2);
  EXPECT_EQ(expanded_tensor3.ndim(), tensor3.ndim() + 1);
  EXPECT_EQ(expanded_tensor3.size(), tensor3.size());
  EXPECT_THAT(expanded_tensor3.dims(), ElementsAre(3, 1, 4, 1, 6));
  EXPECT_THAT(expanded_tensor3.layout().minor_to_major(),
              ElementsAre(4, 3, 2, 1, 0));
  EXPECT_EQ(&tensor3.impl().storage().impl(),
            &expanded_tensor3.impl().storage().impl());
}

TEST(OutOfPlaceSqueezeTest, WithDefaultLayout) {
  Tensor vector2 = hice::full({1}, 1, dtype(kFloat).device(kCPU));
  Tensor squeezed_vector2 = hice::squeeze(vector2, 0);
  EXPECT_EQ(squeezed_vector2.ndim(), vector2.ndim() - 1);
  EXPECT_EQ(squeezed_vector2.size(), vector2.size());
  EXPECT_THAT(squeezed_vector2.dims(), IsEmpty());
  EXPECT_THAT(squeezed_vector2.layout().minor_to_major(), IsEmpty());
  EXPECT_EQ(&vector2.impl().storage().impl(),
            &squeezed_vector2.impl().storage().impl());

  Tensor vector = hice::full({1}, 1, dtype(kFloat).device(kCPU));
  Tensor squeezed_vector = hice::squeeze(vector, -1);
  EXPECT_EQ(squeezed_vector.ndim(), vector.ndim() - 1);
  EXPECT_EQ(squeezed_vector.size(), vector.size());
  EXPECT_THAT(squeezed_vector.dims(), IsEmpty());
  EXPECT_THAT(squeezed_vector.layout().minor_to_major(), IsEmpty());
  EXPECT_EQ(&vector.impl().storage().impl(),
            &squeezed_vector.impl().storage().impl());

  Tensor matrix = hice::full({3, 1}, 1, dtype(kFloat).device(kCPU));
  Tensor squeezed_matrix = hice::squeeze(matrix, 1);
  EXPECT_EQ(squeezed_matrix.ndim(), matrix.ndim() - 1);
  EXPECT_EQ(squeezed_matrix.size(), matrix.size());
  EXPECT_THAT(squeezed_matrix.dims(), ElementsAre(3));
  EXPECT_THAT(squeezed_matrix.layout().minor_to_major(), ElementsAre(0));
  EXPECT_EQ(&matrix.impl().storage().impl(),
            &squeezed_matrix.impl().storage().impl());

  Tensor matrix2 = hice::full({1, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor squeezed_matrix2 = hice::squeeze(matrix2, 0);
  EXPECT_EQ(squeezed_matrix2.ndim(), matrix2.ndim() - 1);
  EXPECT_EQ(squeezed_matrix2.size(), matrix2.size());
  EXPECT_THAT(squeezed_matrix2.dims(), ElementsAre(4));
  EXPECT_THAT(squeezed_matrix2.layout().minor_to_major(), ElementsAre(0));
  EXPECT_EQ(&matrix2.impl().storage().impl(),
            &squeezed_matrix2.impl().storage().impl());

  Tensor matrix3 = hice::full({1, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor squeezed_matrix3 = hice::squeeze(matrix3, -2);
  EXPECT_EQ(squeezed_matrix3.ndim(), matrix3.ndim() - 1);
  EXPECT_EQ(squeezed_matrix3.size(), matrix3.size());
  EXPECT_THAT(squeezed_matrix3.dims(), ElementsAre(4));
  EXPECT_THAT(squeezed_matrix3.layout().minor_to_major(), ElementsAre(0));
  EXPECT_EQ(&matrix3.impl().storage().impl(),
            &squeezed_matrix3.impl().storage().impl());

  Tensor tensor = hice::full({3, 4, 6, 1}, 1, dtype(kFloat).device(kCPU));
  Tensor squeezed_tensor = hice::squeeze(tensor, -1);
  EXPECT_EQ(squeezed_tensor.ndim(), tensor.ndim() - 1);
  EXPECT_EQ(squeezed_tensor.size(), tensor.size());
  EXPECT_THAT(squeezed_tensor.dims(), ElementsAre(3, 4, 6));
  EXPECT_THAT(squeezed_tensor.layout().minor_to_major(), ElementsAre(2, 1, 0));
  EXPECT_EQ(&tensor.impl().storage().impl(),
            &squeezed_tensor.impl().storage().impl());

  Tensor tensor2 = hice::full({3, 1, 4, 6}, 1, dtype(kFloat).device(kCPU));
  Tensor squeezed_tensor2 = hice::squeeze(tensor2, 1);
  EXPECT_EQ(squeezed_tensor2.ndim(), tensor2.ndim() - 1);
  EXPECT_EQ(squeezed_tensor2.size(), tensor2.size());
  EXPECT_THAT(squeezed_tensor2.dims(), ElementsAre(3, 4, 6));
  EXPECT_THAT(squeezed_tensor2.layout().minor_to_major(), ElementsAre(2, 1, 0));
  EXPECT_EQ(&tensor2.impl().storage().impl(),
            &squeezed_tensor2.impl().storage().impl());
}

TEST(InPlaceReshapeTest, WithDefaultLayout) {
  Tensor scalar = hice::full({}, 1, dtype(kFloat).device(kCPU));
  Tensor &reshaped_scalar = hice::reshape_(scalar, {});
  EXPECT_EQ(scalar.ndim(), 0);
  EXPECT_EQ(scalar.size(), 1);
  EXPECT_THAT(scalar.dims(), IsEmpty());
  EXPECT_THAT(scalar.layout().minor_to_major(), IsEmpty());
  EXPECT_EQ(&scalar.impl().storage().impl(),
            &reshaped_scalar.impl().storage().impl());

  Tensor vector = hice::full({2}, 1, dtype(kFloat).device(kCPU));
  Tensor &reshaped_vector = hice::reshape_(vector, {1, 2});
  EXPECT_EQ(vector.ndim(), 2);
  EXPECT_EQ(vector.size(), 2);
  EXPECT_THAT(vector.dims(), ElementsAre(1, 2));
  EXPECT_THAT(vector.layout().minor_to_major(), ElementsAre(1, 0));
  EXPECT_EQ(&vector.impl().storage().impl(),
            &reshaped_vector.impl().storage().impl());

  Tensor matrix = hice::full({3, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor &reshaped_matrix = hice::reshape_(matrix, {3, 1, 4});
  EXPECT_EQ(matrix.ndim(), 3);
  EXPECT_EQ(matrix.size(), 12);
  EXPECT_THAT(matrix.dims(), ElementsAre(3, 1, 4));
  EXPECT_THAT(matrix.layout().minor_to_major(), ElementsAre(2, 1, 0));
  EXPECT_EQ(&matrix.impl().storage().impl(),
            &reshaped_matrix.impl().storage().impl());

  Tensor matrix2 = hice::full({24, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor &reshaped_matrix2 = hice::reshape_(matrix2, {3, 8, 4});
  EXPECT_EQ(matrix2.ndim(), 3);
  EXPECT_EQ(matrix2.size(), 96);
  EXPECT_THAT(matrix2.dims(), ElementsAre(3, 8, 4));
  EXPECT_THAT(matrix2.layout().minor_to_major(), ElementsAre(2, 1, 0));
  EXPECT_EQ(&matrix2.impl().storage().impl(),
            &reshaped_matrix2.impl().storage().impl());

  Tensor tensor = hice::full({4, 7, 4, 6}, 1, dtype(kFloat).device(kCPU));
  Tensor &reshaped_tensor = hice::reshape_(tensor, {28, 24});
  EXPECT_EQ(tensor.ndim(), 2);
  EXPECT_EQ(tensor.size(), 672);
  EXPECT_THAT(tensor.dims(), ElementsAre(28, 24));
  EXPECT_THAT(tensor.layout().minor_to_major(), ElementsAre(1, 0));
  EXPECT_EQ(&tensor.impl().storage().impl(),
            &reshaped_tensor.impl().storage().impl());

  Tensor tensor2 = hice::full({4, 7, 8, 6}, 1, dtype(kFloat).device(kCPU));
  Tensor &reshaped_tensor2 = hice::reshape_(tensor2, {32, 42});
  EXPECT_EQ(tensor2.ndim(), 2);
  EXPECT_EQ(tensor2.size(), 1344);
  EXPECT_THAT(tensor2.dims(), ElementsAre(32, 42));
  EXPECT_THAT(tensor2.layout().minor_to_major(), ElementsAre(1, 0));
  EXPECT_EQ(&tensor2.impl().storage().impl(),
            &reshaped_tensor2.impl().storage().impl());
}

TEST(InPlaceExpandDimsTest, WithDefaultLayout) {
  Tensor scalar = hice::full({}, 1, dtype(kFloat).device(kCPU));
  Tensor &expanded_scalar = hice::expand_dims_(scalar, 0);
  EXPECT_EQ(scalar.ndim(), 1);
  EXPECT_EQ(scalar.size(), 1);
  EXPECT_THAT(scalar.dims(), ElementsAre(1));
  EXPECT_THAT(scalar.layout().minor_to_major(), ElementsAre(0));
  EXPECT_EQ(&scalar.impl().storage().impl(),
            &expanded_scalar.impl().storage().impl());

  Tensor scalar2 = hice::full({}, 1, dtype(kFloat).device(kCPU));
  Tensor &expanded_scalar2 = hice::expand_dims_(scalar2, -1);
  EXPECT_EQ(scalar2.ndim(), 1);
  EXPECT_EQ(scalar2.size(), 1);
  EXPECT_THAT(scalar2.dims(), ElementsAre(1));
  EXPECT_THAT(scalar2.layout().minor_to_major(), ElementsAre(0));
  EXPECT_EQ(&scalar2.impl().storage().impl(),
            &expanded_scalar2.impl().storage().impl());

  Tensor vector = hice::full({2}, 1, dtype(kFloat).device(kCPU));
  Tensor &expanded_vector = hice::expand_dims_(vector, 0);
  EXPECT_EQ(vector.ndim(), 2);
  EXPECT_EQ(vector.size(), 2);
  EXPECT_THAT(vector.dims(), ElementsAre(1, 2));
  EXPECT_THAT(vector.layout().minor_to_major(), ElementsAre(1, 0));
  EXPECT_EQ(&vector.impl().storage().impl(),
            &expanded_vector.impl().storage().impl());

  Tensor vector2 = hice::full({2}, 1, dtype(kFloat).device(kCPU));
  Tensor &expanded_vector2 = hice::expand_dims_(vector2, 1);
  EXPECT_EQ(vector2.ndim(), 2);
  EXPECT_EQ(vector2.size(), 2);
  EXPECT_THAT(vector2.dims(), ElementsAre(2, 1));
  EXPECT_THAT(vector2.layout().minor_to_major(), ElementsAre(1, 0));
  EXPECT_EQ(&vector2.impl().storage().impl(),
            &expanded_vector2.impl().storage().impl());

  Tensor matrix = hice::full({3, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor &expanded_matrix = hice::expand_dims_(matrix, 1);
  EXPECT_EQ(matrix.ndim(), 3);
  EXPECT_EQ(matrix.size(), 12);
  EXPECT_THAT(matrix.dims(), ElementsAre(3, 1, 4));
  EXPECT_THAT(matrix.layout().minor_to_major(), ElementsAre(2, 1, 0));
  EXPECT_EQ(&matrix.impl().storage().impl(),
            &expanded_matrix.impl().storage().impl());

  Tensor matrix2 = hice::full({3, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor &expanded_matrix2 = hice::expand_dims_(matrix2, -3);
  EXPECT_EQ(matrix2.ndim(), 3);
  EXPECT_EQ(matrix2.size(), 12);
  EXPECT_THAT(matrix2.dims(), ElementsAre(1, 3, 4));
  EXPECT_THAT(matrix2.layout().minor_to_major(), ElementsAre(2, 1, 0));
  EXPECT_EQ(&matrix2.impl().storage().impl(),
            &expanded_matrix2.impl().storage().impl());

  Tensor matrix3 = hice::full({3, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor &expanded_matrix3 = hice::expand_dims_(matrix3, -1);
  EXPECT_EQ(matrix3.ndim(), 3);
  EXPECT_EQ(matrix3.size(), 12);
  EXPECT_THAT(matrix3.dims(), ElementsAre(3, 4, 1));
  EXPECT_THAT(matrix3.layout().minor_to_major(), ElementsAre(2, 1, 0));
  EXPECT_EQ(&matrix3.impl().storage().impl(),
            &expanded_matrix3.impl().storage().impl());

  Tensor tensor = hice::full({3, 1, 4, 6}, 1, dtype(kFloat).device(kCPU));
  Tensor &expanded_tensor = hice::expand_dims_(tensor, 0);
  EXPECT_EQ(tensor.ndim(), 5);
  EXPECT_EQ(tensor.size(), 72);
  EXPECT_THAT(tensor.dims(), ElementsAre(1, 3, 1, 4, 6));
  EXPECT_THAT(tensor.layout().minor_to_major(),
              ElementsAre(4, 3, 2, 1, 0));
  EXPECT_EQ(&tensor.impl().storage().impl(),
            &expanded_tensor.impl().storage().impl());

  Tensor tensor2 = hice::full({3, 1, 4, 6}, 1, dtype(kFloat).device(kCPU));
  Tensor &expanded_tensor2 = hice::expand_dims_(tensor2, -1);
  EXPECT_EQ(tensor2.ndim(), 5);
  EXPECT_EQ(tensor2.size(), 72);
  EXPECT_THAT(tensor2.dims(), ElementsAre(3, 1, 4, 6, 1));
  EXPECT_THAT(tensor2.layout().minor_to_major(),
              ElementsAre(4, 3, 2, 1, 0));
  EXPECT_EQ(&tensor2.impl().storage().impl(),
            &expanded_tensor2.impl().storage().impl());

  Tensor tensor3 = hice::full({3, 1, 4, 6}, 1, dtype(kFloat).device(kCPU));
  Tensor &expanded_tensor3 = hice::expand_dims_(tensor3, -2);
  EXPECT_EQ(tensor3.ndim(), 5);
  EXPECT_EQ(tensor3.size(), 72);
  EXPECT_THAT(tensor3.dims(), ElementsAre(3, 1, 4, 1, 6));
  EXPECT_THAT(tensor3.layout().minor_to_major(),
              ElementsAre(4, 3, 2, 1, 0));
  EXPECT_EQ(&tensor3.impl().storage().impl(),
            &expanded_tensor3.impl().storage().impl());
}

TEST(InPlaceSqueezeTest, WithDefaultLayout) {
  Tensor vector2 = hice::full({1}, 1, dtype(kFloat).device(kCPU));
  Tensor &squeezed_vector2 = hice::squeeze_(vector2, 0);
  EXPECT_EQ(vector2.ndim(), 0);
  EXPECT_EQ(vector2.size(), 1);
  EXPECT_THAT(vector2.dims(), IsEmpty());
  EXPECT_THAT(vector2.layout().minor_to_major(), IsEmpty());
  EXPECT_EQ(&vector2.impl().storage().impl(),
            &squeezed_vector2.impl().storage().impl());

  Tensor vector = hice::full({1}, 1, dtype(kFloat).device(kCPU));
  Tensor &squeezed_vector = hice::squeeze_(vector, -1);
  EXPECT_EQ(vector.ndim(), 0);
  EXPECT_EQ(vector.size(), 1);
  EXPECT_THAT(vector.dims(), IsEmpty());
  EXPECT_THAT(vector.layout().minor_to_major(), IsEmpty());
  EXPECT_EQ(&vector.impl().storage().impl(),
            &squeezed_vector.impl().storage().impl());

  Tensor matrix = hice::full({3, 1}, 1, dtype(kFloat).device(kCPU));
  Tensor &squeezed_matrix = hice::squeeze_(matrix, 1);
  EXPECT_EQ(matrix.ndim(), 1);
  EXPECT_EQ(matrix.size(), 3);
  EXPECT_THAT(matrix.dims(), ElementsAre(3));
  EXPECT_THAT(matrix.layout().minor_to_major(), ElementsAre(0));
  EXPECT_EQ(&matrix.impl().storage().impl(),
            &squeezed_matrix.impl().storage().impl());

  Tensor matrix2 = hice::full({1, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor &squeezed_matrix2 = hice::squeeze_(matrix2, 0);
  EXPECT_EQ(matrix2.ndim(), 1);
  EXPECT_EQ(matrix2.size(), 4);
  EXPECT_THAT(matrix2.dims(), ElementsAre(4));
  EXPECT_THAT(matrix2.layout().minor_to_major(), ElementsAre(0));
  EXPECT_EQ(&matrix2.impl().storage().impl(),
            &squeezed_matrix2.impl().storage().impl());

  Tensor matrix3 = hice::full({1, 4}, 1, dtype(kFloat).device(kCPU));
  Tensor &squeezed_matrix3 = hice::squeeze_(matrix3, -2);
  EXPECT_EQ(matrix3.ndim(), 1);
  EXPECT_EQ(matrix3.size(), 4);
  EXPECT_THAT(matrix3.dims(), ElementsAre(4));
  EXPECT_THAT(matrix3.layout().minor_to_major(), ElementsAre(0));
  EXPECT_EQ(&matrix3.impl().storage().impl(),
            &squeezed_matrix3.impl().storage().impl());

  Tensor tensor = hice::full({3, 4, 6, 1}, 1, dtype(kFloat).device(kCPU));
  Tensor &squeezed_tensor = hice::squeeze_(tensor, -1);
  EXPECT_EQ(tensor.ndim(), 3);
  EXPECT_EQ(tensor.size(), 72);
  EXPECT_THAT(tensor.dims(), ElementsAre(3, 4, 6));
  EXPECT_THAT(tensor.layout().minor_to_major(), ElementsAre(2, 1, 0));
  EXPECT_EQ(&tensor.impl().storage().impl(),
            &squeezed_tensor.impl().storage().impl());

  Tensor tensor2 = hice::full({3, 1, 4, 6}, 1, dtype(kFloat).device(kCPU));
  Tensor &squeezed_tensor2 = hice::squeeze_(tensor2, 1);
  EXPECT_EQ(tensor2.ndim(), 3);
  EXPECT_EQ(tensor2.size(), 72);
  EXPECT_THAT(tensor2.dims(), ElementsAre(3, 4, 6));
  EXPECT_THAT(tensor2.layout().minor_to_major(), ElementsAre(2, 1, 0));
  EXPECT_EQ(&tensor2.impl().storage().impl(),
            &squeezed_tensor2.impl().storage().impl());
}

}  // namespace
}  // namespace hice