#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hice/core/index_util.h"
#include "hice/core/shape_util.h"
#include "hice/util/container.h"
#include "hice/util/string_ops.h"
#include "hice/util/types.h"

namespace hice {

bool ShapeUtil::equal(const Shape& lhs, const Shape& rhs) { return lhs == rhs; }

bool ShapeUtil::same_dimensions(const Shape& lhs, const Shape& rhs) {
  return c_equal(lhs.dimensions(), rhs.dimensions());
}

bool ShapeUtil::compatible(const Shape& lhs, const Shape& rhs) {
  return ShapeUtil::same_dimensions(lhs, rhs);
}

int64_t ShapeUtil::get_true_rank(const Shape& shape) {
  int64_t accum = 0;
  for (int64_t dimension : shape.dimensions()) {
    // We do not count zero dimensions.
    if (dimension != 1) {
      accum += 1;
    }
  }
  return accum;
}

int64_t ShapeUtil::get_num_items(const Shape& shape) {
  HICE_DCHECK_EQ(shape.dimensions_size(), shape.rank());
  if (shape.dimensions().size() == 1) {
    return shape.dimensions()[0];
  }
  return std::accumulate<decltype(shape.dimensions().begin()), int64_t>(
      shape.dimensions().begin(), shape.dimensions().end(), 1LL,
      std::multiplies<int64_t>());
}

bool ShapeUtil::is_zero_item(const Shape& shape) {
  return get_num_items(shape) == 0;
}

bool ShapeUtil::is_scalar(const Shape& shape) { return shape.rank() == 0; }

int64_t ShapeUtil::get_dimension(const Shape& shape, int64_t dimension_number) {
  return shape.dimensions(get_dimension_number(shape, dimension_number));
}

int64_t ShapeUtil::get_dimension_number(const Shape& shape,
                                        int64_t dimension_number) {
  if (dimension_number < 0) {
    dimension_number += shape.rank();
  }
  HICE_CHECK_GE(dimension_number, 0);
  return dimension_number;
}

Shape ShapeUtil::delete_dimension(int64_t dim_to_delete, Shape shape) {
  shape.delete_dimension(dim_to_delete);
  return shape;
}

Shape ShapeUtil::filter_dimensions(const std::function<bool(int64_t)>& p,
                                   Shape shape) {
  std::vector<int64_t> dims_to_delete;
  for (int64_t i = shape.dimensions().size() - 1; i >= 0; --i) {
    if (!p(i)) {
      dims_to_delete.push_back(i);
    }
  }
  for (int64_t dim : dims_to_delete) {
    shape = delete_dimension(dim, shape);
  }
  return shape;
}

bool ShapeUtil::has_degenerate_dimensions(const Shape& shape) {
  return c_linear_search(shape.dimensions(), 1);
}

Shape ShapeUtil::delete_degenerate_dimensions(const Shape& shape) {
  return filter_dimensions(
      [&](int64_t dim) -> bool { return shape.dimensions()[dim] != 1; }, shape);
}

std::string ShapeUtil::human_string(const Shape& shape) {
  std::vector<std::string> dim_elements;
  for (int i = 0; i < shape.dimensions_size(); ++i) {
    dim_elements.push_back(StrCat(shape.dimensions(i)));
  }
  return StrCat("[", StrJoin(dim_elements, ","), "]");
}

std::string ShapeUtil::human_string_with_layout(const Shape& shape) {
  std::string result = "[";
  for (int i = 0; i < shape.dimensions().size(); i++) {
    StrAppend(&result, (i > 0) ? "," : "", shape.dimensions(i));
  }
  result += "]";
  if (!is_scalar(shape) && LayoutUtil::has_layout(shape)) {
    StrAppend(&result, LayoutUtil::human_string(shape.layout()));
  }
  return result;
}

Shape ShapeUtil::make_shape(ConstIntArrayRef dimensions) {
  Shape result;
  HICE_CHECK(populate_shape(dimensions, result));
  return result;
}

bool ShapeUtil::populate_shape(ConstIntArrayRef dimensions, Shape& shape) {
  shape.clear();
  for (int64_t dimension : dimensions) {
    shape.add_dimensions(dimension);
  }
  LayoutUtil::set_to_default_layout(shape);
  return validate_shape(shape);
}

// Constructs and returns the new shape with the given minor_to_major order in
// its Layout.
Shape ShapeUtil::make_shape_with_layout(ConstIntArrayRef dimensions,
                                        ConstIntArrayRef minor_to_major) {
  if (dimensions.size() != minor_to_major.size()) {
    HICE_LOG(ERROR) << "Dimensions size is " << dimensions.size()
                    << " but layout size is " << minor_to_major.size();
  }
  Shape shape = ShapeUtil::make_shape(dimensions);
  auto& min2maj = shape.mutable_layout().mutable_minor_to_major();
  min2maj.clear();
  for (int64_t value : minor_to_major) {
    min2maj.push_back(value);
  }
  HICE_CHECK(ShapeUtil::validate_shape(shape));
  return shape;
}

Shape ShapeUtil::make_shape_with_descending_layout(
    ConstIntArrayRef dimensions) {
  std::vector<int64_t> layout(dimensions.size());
  std::iota(layout.rbegin(), layout.rend(), static_cast<int64_t>(0));
  return make_shape_with_layout(dimensions, layout);
}

Shape ShapeUtil::make_shape_with_descending_layout_and_same_physical_layout(
    const Shape& shape) {
  std::vector<int64_t> dims(shape.dimensions_size());
  for (int i = 0; i < shape.dimensions_size(); ++i) {
    dims[i] = shape.dimensions(LayoutUtil::Major(shape.layout(), i));
  }
  return make_shape_with_descending_layout(dims);
}

bool ShapeUtil::validate_shape_with_optional_layout(const Shape& shape) {
  bool is_valid = validate_shape_size_with_optional_layout_internal(shape);
  if (!is_valid) return false;
  return LayoutUtil::validate_layout_in_shape(shape,
                                              /*allow_missing_layouts=*/true);
}

bool ShapeUtil::validate_shape(const Shape& shape) {
  bool is_valid = validate_shape_size_with_optional_layout_internal(shape);
  if (!is_valid) return false;
  return LayoutUtil::validate_layout_in_shape(shape, false);
}

bool ShapeUtil::validate_shape_size_with_optional_layout_internal(
    const Shape& shape) {
  for (int64_t i = 0; i < shape.rank(); ++i) {
    int64_t dimension = shape.dimensions(i);
    if (dimension < 0) {
      return false;
    }
  }
  return validate_shape_size(shape);
}

bool ShapeUtil::validate_shape_size(const Shape& shape) {
  // We can only reason about some aspects of array's shape if it has a valid
  // layout, these aspects will be ignored otherwise.
  // bool shape_has_valid_layout = LayoutUtil::has_layout(shape) &&
  //                              LayoutUtil::validate_layout_in_shape(shape);

  int64_t shape_size = [&]() {
    // Right now we just consider the dense shape
    int64_t dense_shape_size = 1;
    if (shape.dimensions().empty()) {
      return dense_shape_size;
    }

    ConstIntArrayRef shape_max_dimensions = as_int64_slice(shape.dimensions());
    for (int64_t dim : shape_max_dimensions) {
      dense_shape_size = multiply_without_overflow(dense_shape_size, dim);
      if (dense_shape_size < 0) {
        return dense_shape_size;
      }
    }
    return dense_shape_size;
  }();

  // Shape size may overflow int64_t
  if (shape_size < 0) {
    return false;
  }
  return true;
}

Shape ShapeUtil::permute_dimensions(ConstIntArrayRef permutation,
                                    const Shape& shape) {
  Shape new_shape = shape;
  new_shape.clear_dimensions();
  for (auto dim : permute(permutation, shape.dimensions())) {
    new_shape.add_dimensions(dim);
  }
  // If `shape` has a layout, by contract we choose a new layout such that the
  // transpose defined by this permutation is a bitcast.
  //
  // Some formalism helps to understand the correct way to do this.  We're going
  // to do algebra in the group of permutations of the dimensions of `shape`.
  //
  // Since the order of `shape`'s dimensions is not permuted relative to itself,
  // `shape`'s list of dimensions is isomorphic to the identity I.
  //
  // Let `shape`'s layout be L.  A layout is a permutation which maps a
  // minor-to-major physical layout to the order of a shape's logical dims.
  // Therefore inverse of a layout maps from logical to physical dims, and so
  // the physical layout of I is simply L'.I = L', where L' is the inverse of L.
  //
  // Let the argument `permutation` be P.  This is a permutation over `shape`'s
  // dimensions, so our return value will be a shape with dims P.I = P.  Our
  // goal is to construct a layout permutation L* that we can apply to P such
  // that the physical dimension ordering of the returned shape is the same
  // as that of the original shape, namely L'.
  //
  // Our returned shape has dims P and layout L*, so its in-memory layout is
  // L*'.P.  Setting this equal to L' and solving for L*, we get:
  //
  //   L*'.P = L'    =>
  //   L*'   = L'P'  =>
  //   L*    = P.L
  //
  if (shape.has_layout()) {
    HICE_CHECK(LayoutUtil::is_dense(shape));
    Layout& new_layout = new_shape.mutable_layout();
    new_layout.set_type(kDense);
    new_layout.clear_minor_to_major();
    for (auto index : compose_permutations(
             permutation, as_int64_slice(shape.layout().minor_to_major()))) {
      new_layout.add_minor_to_major(index);
    }
    // The permutation accepted by transpose_is_bitcast is the inverse of the
    // permutation here.
    HICE_CHECK(transpose_is_bitcast(shape, new_shape,
                                    inverse_permutation(permutation)))
        << "shape=" << human_string_with_layout(shape)
        << ", new_shape=" << human_string_with_layout(new_shape)
        << ", permutation={" << StrJoin(permutation, ",") << "}";
  }
  return new_shape;
}

std::tuple<bool, std::vector<int64_t>, std::vector<int64_t>>
ShapeUtil::inserted_or_deleted_1sized_dimensions(const Shape& shape_pre,
                                                 const Shape& shape_post) {
  auto nil =
      std::make_tuple(false, std::vector<int64_t>(), std::vector<int64_t>());

  std::vector<int64_t> deleted_indices;
  std::vector<int64_t> inserted_indices;
  // Returns false if any input/output index between prior_unmodified_dim_pair
  // and unmodified_dim_pair have size >1. Otherwise, returns true and appends
  // the degerenate input/output dimensions in the gap to
  // deleted_indices/inserted_indices respectively.
  auto check_modified_dims =
      [&shape_pre, &shape_post, &deleted_indices, &inserted_indices](
          std::pair<int64_t, int64_t> prior_unmodified_dim_pair,
          std::pair<int64_t, int64_t> unmodified_dim_pair) {
        for (int64_t modified_input_dim = prior_unmodified_dim_pair.first + 1;
             modified_input_dim < unmodified_dim_pair.first;
             ++modified_input_dim) {
          if (shape_pre.dimensions(modified_input_dim) > 1) {
            return false;
          }
          deleted_indices.push_back(modified_input_dim);
        }
        for (int64_t modified_output_dim = prior_unmodified_dim_pair.second + 1;
             modified_output_dim < unmodified_dim_pair.second;
             ++modified_output_dim) {
          if (shape_post.dimensions(modified_output_dim) > 1) {
            return false;
          }
          inserted_indices.push_back(modified_output_dim);
        }
        return true;
      };

  std::vector<std::pair<int64_t, int64_t>> unmodified_dims =
      dimensions_unmodified_by_reshape(shape_pre, shape_post);
  // Returns nil if the reshape modifies any non-degenerate input/output
  // dimension. dimensions_unmodified_by_reshape gives us all unmodified
  // dimensions, so we only need to check whether dimensions in the gaps (thus
  // modified) have size >1.
  for (size_t i = 0; i <= unmodified_dims.size(); ++i) {
    // Check (modified) dimensions between unmodified_dims[i-1] and
    // unmodified_dims[i].
    auto prior_unmodified_dim_pair =
        i > 0 ? unmodified_dims[i - 1] : std::make_pair(-1L, -1L);
    auto unmodified_dim_pair =
        i < unmodified_dims.size()
            ? unmodified_dims[i]
            : std::make_pair(shape_pre.rank(), shape_post.rank());
    if (!check_modified_dims(prior_unmodified_dim_pair, unmodified_dim_pair)) {
      return nil;
    }
  }

  return std::make_tuple(true, deleted_indices, inserted_indices);
}

std::vector<std::pair<int64_t, int64_t>>
ShapeUtil::dimensions_unmodified_by_reshape(const Shape& input_shape,
                                            const Shape& output_shape) {
  // Unmodified dimensions are merely common factors of rank 1.
  auto common_factors =
      hice::common_factors(as_int64_slice(input_shape.dimensions()),
                           as_int64_slice(output_shape.dimensions()));
  for (size_t i = 0; i < common_factors.size() - 1;) {
    if (1 != common_factors[i + 1].first - common_factors[i].first ||
        1 != common_factors[i + 1].second - common_factors[i].second) {
      common_factors.erase(common_factors.begin() + i);
    } else {
      ++i;
    }
  }
  // `common_factors(a, b).back() == (a.rank, b.rank)` so we must pop it.
  common_factors.pop_back();
  return common_factors;
}

bool ShapeUtil::transpose_is_bitcast(const Shape& input_shape,
                                     const Shape& output_shape,
                                     ConstIntArrayRef dimension_mapping) {
  // Check the reshape permutes the positions of each dimension in the
  // minor-to-major order. positions[i]=k means dimension `i` is k-th minor.
  //   input_positions = apply(dimension_mapping, output_positions)
  //
  // Because the positions of each dimension are the inverse permutation of the
  // minor-to-major order, the above check is equivalent to
  //   inverse(input_dimensions) =
  //       apply(dimension_mapping, inverse(output_dimensions))
  //   # `I` indicates identity permutation.
  //   apply(input_dimensions, I) =
  //       apply(dimension_mapping, apply(output_dimensions, I))
  //   apply(input_dimensions, I) =
  //       apply((dimension_mapping * output_dimensions), I)
  //   input_dimensions = dimension_mapping * output_dimensions
  return c_equal(compose_permutations(
                     dimension_mapping,
                     as_int64_slice(output_shape.layout().minor_to_major())),
                 input_shape.layout().minor_to_major());
}

bool ShapeUtil::reshape_is_bitcast(const Shape& input_shape,
                                   const Shape& output_shape) {
  CHECK_EQ(get_num_items(input_shape), get_num_items(output_shape));
  if (get_num_items(input_shape) == 0) {
    return true;
  }

  // TL;DR: The rest of the method checks that the reshape does not change the
  // physical location of any unit input or output index. Unit indices have
  // exactly one dimension that equals 1 and other dimensions 0. This condition
  // is necessary for the reshape to be a bitcast, because a bitcast-equivalent
  // reshape shouldn't change the physical location of any element. It is also a
  // sufficient condition as is proved below (note: many details are omitted for
  // space).
  //
  // Definitions:
  //
  // * Denote the input shape by IS and output shape by OS. IS[i] or OS[i] means
  // the size of i-th least significant dimension of IS or OS (this is opposite
  // to how we define the index of Shape::dimensions()).
  //
  // * Given an input or output index I, denote by p(I) I's physical linear
  // index (or physical index for short) and l(I) I's logical linear index (or
  // logical index for short).
  //
  // * Given a logical index k, denote by II(k) the input index whose linear
  // index is k, and OI(k) the corresponding output index.
  //
  // * Denote by IT[i] the increment of physical index if i-th dimension of the
  // input index is increased by 1. Similarly, OT[i] means the increment if i-th
  // dimension of the output index is increased by 1. Note that IT[i] or OT[i]
  // is a function of IS or OS and the layout, and not dependent on the specific
  // input or output index.
  //
  // To prove the reshape from IS to OS is a bitcast, it is sufficient to prove
  // that, for any linear index k, p(II(k))=p(OI(k)). We prove this by
  // induction. We know p(II(0))=p(OI(0)) is trivially true, so what's left is
  // to prove, with every increment on k, the above formula still holds.
  //
  // First, suppose reshaping from IS to OS is non-factorizable (we discuss
  // refactorizable reshapes later). A reshape from IS to OS is factorizable, if
  // there exists (i,j) such that
  //
  //   0<=i<=|IS|
  //   0<=j<=|OS|
  //   |IS|-i+|OS|-j > 0 (i.e., i,j mustn't both point to the end)
  //   product(IS[i], IS[i+1], ..., IS[|IS|-1])
  //     = product(OS[j], OS[j+1], ..., OS[|OS|-1])
  //
  // p(II(k))=p(OI(k)) is trivially true for k=0 because p(II(0)) and p(OI(0))
  // are both 0. It's also trivially true for k=1, because II(1) and OI(1) are
  // unit indices which are already tested. This also means IT[0]=OT[0]
  // because p(II(1))=IT[0] and p(OI(1))=OT[0].
  //
  // Furthermore, p(II(k))=p(OI(k)) for k<min(IS[0],OS[0]), because each
  // increment of k adds IT[0] to the input physical and OT[0] (same as IT[0])
  // to the output physical.
  //
  // When k=min(IS[0],OS[0]), the first wrap happens. Without losing generality,
  // suppose IS[0]<OS[0] and thus k=IS[0]. Similar proof applies to IS[0]>OS[0].
  // Note that IS[0]!=OS[0] because the reshape is non-factorizable. From
  // logical index k-1 to logical index k, dimension 1 of the input index
  // is increased by 1 and dimension 0 is reset to 0 thus decreased by
  // IS[0]-1. Therefore, the physical input index is increased by
  //
  //   p(II(k)) - p(II(k-1)) = IT[1] - (IS[0]-1) * IT[0]
  //
  // Because IS[0]<OS[0], the only change to the output index is that its
  // dimension 0 is increased by one. Therefore,
  //
  //   p(OI(k)) - p(OI(k-1)) = OT[0] = IT[0]
  //
  // Because II(k) is an unit index -- (0,..,0,1,0), we already tested that
  // p(II(k))=p(OI(k)). Therefore,
  //   IT[1] - (IS[0]-1) * IT[0] = IT[0]
  //   IT[1] = IS[0] * IT[0]
  // In other words, input dimension 1 is immediately more major than input
  // dimension 0. We can now conceptually collapse these two dimensions because
  // an increment in the logical index affecting only these two dimensions maps
  // to IT[0] in the physical index.
  //
  // By induction (omitted here), we can prove IT[i]=IS[i-1]*IT[i-1] and
  // OT[i]=OS[i-1]*OT[i-1]. Therefore, both IS and OS are row-major and bitwise
  // identical.
  //
  // A factorizable reshape can be factorized into a list of non-factorizable
  // sub-reshapes, each of which can be handled similarly to the proof above.
  // For example,
  //
  //   [7x9x2x15] -> [63x6x5]
  //
  // can be factorized into
  //
  //   [7x9] -> [63] and [2x15] -> [6x5].
  //
  // Suppose input index I=(x3,x2,x1,x0) and output index O=(y2,y1,y0) have the
  // same logical linear index. According to the factorization, we know
  // l(x3,x2,0,0)=l(y2,0,0) and l(0,0,x1,x0)=l(0,y1,y0). Using the proof for
  // non-factorizable reshapes, we can prove p(0,0,x1,x0)=p(0,y1,y0). Using a
  // similar proof, with the increment of the logical index set to
  // IS[1]*IS[0]=OS[1]*OS[0]=30 instead of 1, we can prove
  // p(x3,x2,0,0)=p(y2,0,0) too. Therefore,
  //
  //   p(x3,x2,x1,x0) = p(x3,x2,0,0) + p(0,0,x1,x0)
  //                  = p(y2,0,0) + p(0,0,y1,y0)
  //                  = p(y2,y1,y0)
  //
  // check_input_unit_indices checks one way of the condition: each input unit
  // index is mapped to an output index with the same physical location. This
  // lambda will be called again with input_shape and output_shape reversed to
  // check the other way.
  auto check_input_unit_indices = [](const Shape& input_shape,
                                     const Shape& output_shape) {
    // input_shape_dim0_major/output_shape_dim0_major has the same "dimensions"
    // as input_shape/output_shape and the dimension-0-major layout. These two
    // shapes are used for conversion between logical linear indices and
    // multi-dimensional indices.
    Shape input_shape_dim0_major = make_shape_with_descending_layout(
        as_int64_slice(input_shape.dimensions()));
    Shape output_shape_dim0_major = make_shape_with_descending_layout(
        as_int64_slice(output_shape.dimensions()));

    for (int64_t input_dim = 0; input_dim < input_shape.rank(); ++input_dim) {
      if (input_shape.dimensions(input_dim) <= 1) {
        continue;
      }

      std::vector<int64_t> input_unit_index(input_shape.rank(), 0);
      input_unit_index[input_dim] = 1;
      int64_t logical_linear_index = IndexUtil::multi_index_to_offset(
          input_shape_dim0_major, input_unit_index);
      // output_index has the same logical linear index as input_unit_index.
      std::vector<int64_t> output_index = IndexUtil::offset_to_multi_index(
          output_shape_dim0_major, logical_linear_index);
      // Check input_unit_index and output_index have the same physical linear
      // index.
      if (IndexUtil::multi_index_to_offset(input_shape, input_unit_index) !=
          IndexUtil::multi_index_to_offset(output_shape, output_index)) {
        return false;
      }
    }
    return true;
  };
  return check_input_unit_indices(input_shape, output_shape) &&
         check_input_unit_indices(output_shape, input_shape);
}

optional<Shape> ShapeUtil::align_layouts(const Shape& input_shape,
                                         const Shape& output_shape) {
  int64_t input_rank = input_shape.rank();
  int64_t output_rank = output_shape.rank();

  // First, calculate an alignment of the dimensions. A consecutive sequence of
  // input dimensions and output dimensions belong to the same alignment part if
  // the products of their dimension bounds are the same. In the easiest case,
  // an alignment part consists of one input dimension and one output dimension
  // which both have the same dimension bound. An alignment part specifies which
  // dimensions need to be kept together in a physical layout if we want a
  // reshape to be a bitcast. The order of the alignment parts is defined by the
  // physical layout of the input shape, so when we construct the layout for the
  // output shape we just process the alignment parts in this order, and then
  // layout the dimensions belonging to each part in descending (major to minor)
  // order.

  // Stores the input and output dimension numbers where each alignment part
  // starts.
  std::vector<std::pair<int64_t, int64_t>> alignment;
  alignment.push_back({0, 0});

  // Stores a mapping from the input dimension to the alignment part it belongs
  // to.
  std::vector<int64_t> dimension_to_alignment_index(input_rank);
  int64_t input_dimension_product = 1, output_dimension_product = 1;
  for (int64_t i = 0, j = 0; i < input_rank || j < output_rank;) {
    // Check if we have reached the end of an alignment part.
    if (input_dimension_product == output_dimension_product &&
        input_dimension_product > 1) {
      alignment.push_back({i, j});
      input_dimension_product = output_dimension_product = 1;
    }
    if (input_dimension_product < output_dimension_product ||
        j == output_rank) {
      if (i == input_rank) {
        return nullopt;
      }
      dimension_to_alignment_index[i] = alignment.size() - 1;
      input_dimension_product *= input_shape.dimensions(i);
      ++i;
    } else {
      output_dimension_product *= output_shape.dimensions(j);
      ++j;
    }
  }
  if (input_dimension_product != output_dimension_product) {
    return nullopt;
  }
  // We also need to store an end element so that we know where the last
  // alignment part ends.
  alignment.push_back({input_rank, output_rank});

  // Now check if the physical layout can potentially be aligned to the output
  // shape by changing the physical layout of the output shape. We need to check
  // that all dimension numbers that belong to the same alignment part appear
  // consecutively, and are in descending order. However we can ignore any
  // trivial dimension bounds of 1, because they can be placed anywhere.
  auto input_dimension_numbers = input_shape.layout().minor_to_major();
  std::vector<int64_t> output_layout;
  output_layout.reserve(output_rank);
  for (int64_t i = 0; i < input_rank;) {
    int64_t current_dimension_number = input_dimension_numbers[i];

    // Skip trivial dimensions with a bound of 1.
    if (input_shape.dimensions(current_dimension_number) == 1) {
      ++i;
      continue;
    }

    // Calculate the number of non-trivial dimension bounds in the input shape
    // belonging to the current alignment part.
    const int64_t current_alignment_index =
        dimension_to_alignment_index[current_dimension_number];
    // Because of the special end element that we added, we can be sure that
    // 'current_alignment_index' is < alignment.size() - 1.
    HICE_CHECK_LT(current_alignment_index, alignment.size() - 1);
    int64_t num_non_trivial_dimensions_in_alignment_part = 0;
    for (int64_t j = alignment[current_alignment_index].first;
         j < alignment[current_alignment_index + 1].first; ++j) {
      if (input_shape.dimensions(j) != 1) {
        ++num_non_trivial_dimensions_in_alignment_part;
      }
    }

    // Check that the following 'num_non_trivial_dimensions_in_alignment_part'
    // dimension numbers (ignoring dimension numbers with dimension bound 1) are
    // in descending order and belong to the current alignment part.
    for (int64_t j = 0; j < num_non_trivial_dimensions_in_alignment_part;
         ++i, ++j) {
      if (i == input_rank) {
        return nullopt;
      }
      // Skip trivial dimensions with a bound of 1.
      if (input_shape.dimensions(input_dimension_numbers[i]) == 1) {
        --j;
        continue;
      }
      // If the current dimension number belongs to a different alignment part,
      // or the dimension numbers are not in descending order, we can return
      // early.
      if (dimension_to_alignment_index[input_dimension_numbers[i]] !=
              current_alignment_index ||
          input_dimension_numbers[i] > current_dimension_number) {
        return nullopt;
      }
      current_dimension_number = input_dimension_numbers[i];
    }

    // The output dimension numbers that belong to the current alignment part
    // need to appear in the same descending order as in the input. Again, we
    // can skip dimensions with a bound of 1.
    for (int64_t j = alignment[current_alignment_index + 1].second - 1;
         j >= alignment[current_alignment_index].second; --j) {
      if (output_shape.dimensions(j) != 1) {
        output_layout.push_back(j);
      }
    }
  }
  // Now add all the dimensions with dimension bound 1 at the end of
  // 'output_layout'.
  for (int64_t i = 0; i < output_rank; ++i) {
    if (output_shape.dimensions(i) == 1) {
      output_layout.push_back(i);
    }
  }
  CHECK_EQ(output_layout.size(), output_rank);
  Shape output_shape_with_layout = make_shape_with_layout(
      as_int64_slice(output_shape.dimensions()), output_layout);
  HICE_CHECK(reshape_is_bitcast(input_shape, output_shape_with_layout))
      << "reshape is not a bitcast for input_shape: "
      << ShapeUtil::human_string_with_layout(input_shape)
      << " and output_shape_with_layout: "
      << ShapeUtil::human_string_with_layout(output_shape_with_layout);
  return output_shape_with_layout;
}

}  // namespace hice