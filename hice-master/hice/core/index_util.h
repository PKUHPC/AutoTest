#pragma once 

#include <vector>
#include <string>

#include "hice/util/types.h"
#include "hice/core/shape.h"
#include "hice/core/macros.h"

namespace hice {

// Namespaced collection of (static) utilities related to indexing into
// multidimensional arrays.
class HICE_API IndexUtil {
 public:
  // Converts a multidimensional index (eg {x, y, z}) into a linear index based
  // on the shape and its layout. The first index in the multi_index is
  // dimension 0.
  static int64_t multi_index_to_offset(const Shape& shape,
                                       ConstIntArrayRef multi_index);

  // Converts a linear index into multidimensional index (eg {x, y, z}) based on
  // the shape and its layout. The first index in the returned multidimensional
  // index is dimension 0.
  static std::vector<int64_t> offset_to_multi_index(const Shape& shape,
                                                    int64_t linear_index);

  // Reset to [d(0) - 1, d(1) - 1, d(2) - 1, ..., d(n-1) - 1] when underflow occurs                                              
  static void last_multi_index(const Shape& shape, std::vector<int64_t>& multi_index);
                                                
  // Reset to [0, 0, 0, ..., 0] when overflow occurs      
  static void next_multi_index(const Shape& shape, std::vector<int64_t>& multi_index);

  // Bumps a sequence of indices; e.g. {0,0,0,0} up by one index value; e.g. to
  // {0,0,0,1}. This is akin to std::next_permutation. If the index hits a limit
  // for the provided shape, the next most significant index is bumped, in a
  // counting-up process.
  //
  // E.g. for shape f32[2,3]
  //  {0,0}=>{0,1}
  //  {0,1}=>{0,2}
  //  {0,2}=>{1,0}
  //  etc.
  //
  // This is useful for traversing the indices in a literal.
  //
  // Returns true iff the indices were successfully bumped; false if we've hit
  // the limit where it can no longer be bumped in-bounds.
  static bool bump_indices(const Shape& shape, ArrayRef<int64_t> indices);

  // Calculates the stride size (in number of elements, not byte size) of a
  // given logical shape dimension (from 0 to rank-1).
  // Example:
  //  GetDimensionStride(F32[5,8,10,4]{3,2,1,0}, 1) ==
  //    sizeof(dimension(3)) * sizeof(dimension(2)) == 4 * 10
  static int64_t get_dim_stride(const Shape& shape, int64_t dimension);

  static std::vector<int64_t> get_all_strides(const Shape& shape);

  // Returns true iff the given multi-index is contained in the bounds for the
  // shape.
  static bool index_in_bounds(const Shape& shape,
                              ConstIntArrayRef index);

  // Compares the given indices in lexicographic order.  lhs[0] and rhs[0] are
  // compared first, and lhs[rank-1] and rhs[rank-1] last.  If lhs is larger,
  // then -1 is returned. If rhs is larger, then 1 is returned.  Otherwise, 0 is
  // returned.
  static int compare_indices(ConstIntArrayRef lhs,
                             ConstIntArrayRef rhs);

 private:
  HICE_DISABLE_COPY_AND_ASSIGN(IndexUtil);
};

}  // namespace hice 
