#include "hice/core/tensor_printer.h"
#include "hice/basic/factories.h"
#include "hice/math/binary_expr.h"
#include "hice/math/unary_expr.h"

#include "test/tools/compare.h"
#include "gtest/gtest.h"
#include "sys/time.h"
namespace hice {

#if 0
// unary
using UnaryOp = std::function<Tensor(Tensor& a)>;
auto EXP  = [](Tensor& a)->Tensor{ return exp(a); };
auto LOG  = [](Tensor& a)->Tensor{ return log(a); };
auto NEG  = [](Tensor& a)->Tensor{ return neg(a); };

template <typename TScalarType>
struct UnaryTestParams {
  std::vector<int64_t> dims_input;
  std::vector<int64_t> dims_output;
  UnaryOp Op;
};

template <typename TScalarType>
class UnaryTest
    : public ::testing::TestWithParam<UnaryTestParams<TScalarType>> {};

using UnaryTestParamsFloat = UnaryTestParams<float>;
using UnaryTestFloat = UnaryTest<float>;
TEST_P(UnaryTestFloat, RandUniformValueTests) {
  UnaryTestParamsFloat params =
      ::testing::TestWithParam<UnaryTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat));   
  Tensor cpu_output = params.Op(cpu_input);
  Tensor cuda_input = cpu_input.to(kCUDA);   
  Tensor cuda_output = params.Op(cuda_input);  
  ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
}

INSTANTIATE_TEST_CASE_P(
    UnaryTestFloatSuite, UnaryTestFloat,
    ::testing::Values(
      // scalar
      UnaryTestParamsFloat{{}, 
                           {},
                            EXP},
      // tensor
      UnaryTestParamsFloat{{500000}, 
                           {500000},
                            EXP},
      UnaryTestParamsFloat{{600000}, 
                           {600000},
                            LOG},
      UnaryTestParamsFloat{{8000, 300}, 
                           {8000, 300},
                            NEG},
      // parallel(size>=2^15)
      UnaryTestParamsFloat{{330000}, 
                           {330000},
                            NEG},
      // // parallel(size>=2^15)
      // UnaryTestParamsFloat{{3300000}, 
      //                      {3300000},
      //                       EXP}
    )
);

#endif

#if 1
// binary Tensor-Tensor
using BinaryTTOp = std::function<Tensor(Tensor& a, Tensor& b)>;
auto ADDTT = [](Tensor& a, Tensor& b)->Tensor{ return add(a, b); };
auto SUBTT = [](Tensor& a, Tensor& b)->Tensor{ return sub(a, b); };
auto MULTT = [](Tensor& a, Tensor& b)->Tensor{ return mul(a, b); };
auto DIVTT = [](Tensor& a, Tensor& b)->Tensor{ return div(a, b); };
auto MAXTT = [](Tensor& a, Tensor& b)->Tensor{ return max(a, b); };

template <typename TScalarType>
struct BinaryTTTestParams {
  std::vector<int64_t> dims_input1;
  std::vector<int64_t> dims_input2;
  std::vector<int64_t> dims_output;
  BinaryTTOp Op;
};

template <typename TScalarType>
class BinaryTTTest
    : public ::testing::TestWithParam<BinaryTTTestParams<TScalarType>> {};

using BinaryTTTestParamsFloat = BinaryTTTestParams<float>;
using BinaryTTTestFloat = BinaryTTTest<float>;
TEST_P(BinaryTTTestFloat, RandUniformValueTests) {
  BinaryTTTestParamsFloat params =
      ::testing::TestWithParam<BinaryTTTestParamsFloat>::GetParam();
  Tensor cpu_input1 = rand_uniform(params.dims_input1, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_input2 = rand_uniform(params.dims_input2, 1.0, 10.0, device(kCPU).dtype(kFloat));
        struct timeval hice_start, hice_end;
        double hice_time;
        gettimeofday(&hice_start,NULL);

        Tensor cpu_output = params.Op(cpu_input1, cpu_input2);
        gettimeofday(&hice_end,NULL);
        hice_time = (hice_end.tv_sec - hice_start.tv_sec) * 1000.0
                    + (hice_end.tv_usec - hice_start.tv_usec) / 1000.0 ;


        std::cout<< /*GREEN <<*/ "\t[ HICE ] " << /*RESET <<*/ hice_time << " ms" << std::endl;
//  Tensor cuda_input1 = cpu_input1.to(kCUDA);
//  Tensor cuda_input2 = cpu_input2.to(kCUDA);
//  Tensor cuda_output = params.Op(cuda_input1, cuda_input2);
//  ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
}

INSTANTIATE_TEST_CASE_P(
    BinaryTTTestFloatSuite, BinaryTTTestFloat,
    ::testing::Values(
      // none-broadcast
      BinaryTTTestParamsFloat{{}, 
                              {}, 
                              {},
                              ADDTT},
      BinaryTTTestParamsFloat{{6}, 
                              {6}, 
                              {6},
                              SUBTT},
      BinaryTTTestParamsFloat{{80000}, 
                              {80000}, 
                              {80000},
                              MULTT},
      BinaryTTTestParamsFloat{{10000}, 
                              {10000}, 
                              {10000},
                              DIVTT},
      BinaryTTTestParamsFloat{{300000}, 
                              {300000}, 
                              {300000},
                              MAXTT},
// #if 0
      // broadcast
      BinaryTTTestParamsFloat{{}, 
                              {5, 6, 7}, 
                              {5, 6, 7},
                              ADDTT},
      BinaryTTTestParamsFloat{{5, 6, 7}, 
                              {}, 
                              {5, 6, 7},
                              ADDTT},
      BinaryTTTestParamsFloat{{1, 6, 7}, 
                              {5, 6, 7}, 
                              {5, 6, 7},
                              ADDTT},
      BinaryTTTestParamsFloat{{1, 1, 1}, 
                              {2, 3, 5}, 
                              {2, 3, 5},
                              SUBTT},
      BinaryTTTestParamsFloat{{1, 6, 1, 4}, 
                              {2, 1, 3, 1}, 
                              {2, 6, 3, 4},
                              MULTT},
      BinaryTTTestParamsFloat{         {2}, 
                              {2, 3, 1, 2}, 
                              {2, 3, 1, 2},
                              DIVTT},
      // broadcast && parallel
      BinaryTTTestParamsFloat{{1, 60, 70}, 
                              {8, 60, 70}, 
                              {8, 60, 70},
                              MAXTT}
// #endif
    )
);



// #if 0
// binary Tensor-Scalar(numerical)
using BinaryTSOp = std::function<Tensor(Tensor& a, Scalar b)>;
auto ADDTS = [](Tensor& a, Scalar b)->Tensor{ return add(a, b); };
auto SUBTS = [](Tensor& a, Scalar b)->Tensor{ return sub(a, b); };
auto MULTS = [](Tensor& a, Scalar b)->Tensor{ return mul(a, b); };
auto DIVTS = [](Tensor& a, Scalar b)->Tensor{ return div(a, b); };
auto MAXTS = [](Tensor& a, Scalar b)->Tensor{ return max(a, b); };

template <typename TScalarType>
struct BinaryTSTestParams {
  std::vector<int64_t> dims_input1;
  std::vector<int64_t> dims_output;
  BinaryTSOp Op;
};

template <typename TScalarType>
class BinaryTSTest
    : public ::testing::TestWithParam<BinaryTSTestParams<TScalarType>> {};

using BinaryTSTestParamsFloat = BinaryTSTestParams<float>;
using BinaryTSTestFloat = BinaryTSTest<float>;
TEST_P(BinaryTSTestFloat, RandUniformValueTests) {
  BinaryTSTestParamsFloat params =
      ::testing::TestWithParam<BinaryTSTestParamsFloat>::GetParam();
  Tensor cpu_input1 = rand_uniform(params.dims_input1, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  float scalar = cpu_input1.data<float>()[0];
        struct timeval hice_start, hice_end;
        double hice_time;
        gettimeofday(&hice_start,NULL);

        Tensor cpu_output = params.Op(cpu_input1, scalar);
        gettimeofday(&hice_end,NULL);
        hice_time = (hice_end.tv_sec - hice_start.tv_sec) * 1000.0
                    + (hice_end.tv_usec - hice_start.tv_usec) / 1000.0 ;


        std::cout<< /*GREEN <<*/ "\t[ HICE ] " << /*RESET <<*/ hice_time << " ms" << std::endl;
//  Tensor cuda_input1 = cpu_input1.to(kCUDA);
//  Tensor cuda_output = params.Op(cuda_input1, scalar);
//  ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    BinaryTSTestFloatSuite, BinaryTSTestFloat,
    ::testing::Values(
      // broadcast
      BinaryTSTestParamsFloat{{5}, 
                              {5},
                              ADDTS},
      BinaryTSTestParamsFloat{{2, 3, 1, 2}, 
                              {2, 3, 1, 2},
                              SUBTS},
      BinaryTSTestParamsFloat{{2, 6, 3, 4}, 
                              {2, 6, 3, 4},
                              MULTS},
      BinaryTSTestParamsFloat{{2, 3, 5}, 
                              {2, 3, 5},
                              DIVTS},
      BinaryTSTestParamsFloat{{5, 6, 7}, 
                              {5, 6, 7},
                              MAXTS},
      // broadcast && parallel
      BinaryTSTestParamsFloat{{8, 60, 70}, 
                              {8, 60, 70},
                              MAXTS}
    )
);




// bianry Scalar-Tensor
using BinarySTOp = std::function<Tensor(Scalar a, Tensor& b)>;
auto ADDST = [](Scalar a, Tensor& b)->Tensor{ return add(a, b); };
auto SUBST = [](Scalar a, Tensor& b)->Tensor{ return sub(a, b); };
auto MULST = [](Scalar a, Tensor& b)->Tensor{ return mul(a, b); };
auto DIVST = [](Scalar a, Tensor& b)->Tensor{ return div(a, b); };
auto MAXST = [](Scalar a, Tensor& b)->Tensor{ return max(a, b); };

template <typename TScalarType>
struct BinarySTTestParams {
  std::vector<int64_t> dims_input2;
  std::vector<int64_t> dims_output;
  BinarySTOp Op;
};

template <typename TScalarType>
class BinarySTTest
    : public ::testing::TestWithParam<BinarySTTestParams<TScalarType>> {};

using BinarySTTestParamsFloat = BinarySTTestParams<float>;
using BinarySTTestFloat = BinarySTTest<float>;
TEST_P(BinarySTTestFloat, RandUniformValueTests) {
  BinarySTTestParamsFloat params =
      ::testing::TestWithParam<BinarySTTestParamsFloat>::GetParam();
  Tensor cpu_input2 = rand_uniform(params.dims_input2, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  float scalar = cpu_input2.data<float>()[0];

        struct timeval hice_start, hice_end;
        double hice_time;
        gettimeofday(&hice_start,NULL);

        Tensor cpu_output = params.Op(scalar, cpu_input2);
        gettimeofday(&hice_end,NULL);
        hice_time = (hice_end.tv_sec - hice_start.tv_sec) * 1000.0
                    + (hice_end.tv_usec - hice_start.tv_usec) / 1000.0 ;


        std::cout<< /*GREEN <<*/ "\t[ HICE ] " << /*RESET <<*/ hice_time << " ms" << std::endl;
//  Tensor cuda_input2 = cpu_input2.to(kCUDA);
//  Tensor cuda_output = params.Op(scalar, cuda_input2);
//  ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    BinarySTTestFloatSuite, BinarySTTestFloat,
    ::testing::Values(
      // broadcast
      BinarySTTestParamsFloat{{5}, 
                              {5},
                              ADDST},
      BinarySTTestParamsFloat{{2, 3, 1, 2}, 
                              {2, 3, 1, 2},
                              SUBST},
      BinarySTTestParamsFloat{{2, 6, 3, 4}, 
                              {2, 6, 3, 4},
                              MULST},
      BinarySTTestParamsFloat{{2, 3, 5}, 
                              {2, 3, 5},
                              DIVST},
      BinarySTTestParamsFloat{{5, 6, 7}, 
                              {5, 6, 7},
                              MAXST},
      // broadcast && parallel
      BinarySTTestParamsFloat{{8, 60, 70}, 
                              {8, 60, 70},
                              MAXST}
    )
);



/*

 Inplace Test



 */

using UnaryInplaceOp = std::function<Tensor(Tensor& a, Tensor& b)>;
auto EXP_INPLACE  = [](Tensor& a, Tensor& b)->Tensor{ return exp(a, b); };
auto LOG_INPLACE  = [](Tensor& a, Tensor& b)->Tensor{ return log(a, b); };
auto NEG_INPLACE  = [](Tensor& a, Tensor& b)->Tensor{ return neg(a, b); };

template <typename TScalarType>
struct UnaryInplaceTestParams {
  std::vector<int64_t> dims_input;
  std::vector<int64_t> dims_output;
  UnaryInplaceOp Op;
};

template <typename TScalarType>
class UnaryInplaceTest
    : public ::testing::TestWithParam<UnaryInplaceTestParams<TScalarType>> {};

using UnaryInplaceTestParamsFloat = UnaryInplaceTestParams<float>;
using UnaryInplaceTestFloat = UnaryInplaceTest<float>;
TEST_P(UnaryInplaceTestFloat, Inplace) {
  UnaryInplaceTestParamsFloat params =
      ::testing::TestWithParam<UnaryInplaceTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
//  Tensor cuda_input = cpu_input.to(kCUDA);
  Tensor cpu_output = rand_uniform(params.dims_output, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
//  Tensor cuda_output = rand_uniform(params.dims_output, 1.0, 10.0, device(kCUDA).dtype(kFloat));

        struct timeval hice_start, hice_end;
        double hice_time;
        gettimeofday(&hice_start,NULL);

        params.Op(cpu_input, cpu_output);
        gettimeofday(&hice_end,NULL);
        hice_time = (hice_end.tv_sec - hice_start.tv_sec) * 1000.0
                    + (hice_end.tv_usec - hice_start.tv_usec) / 1000.0 ;


        std::cout<< /*GREEN <<*/ "\t[ HICE ] " << /*RESET <<*/ hice_time << " ms" << std::endl;
//  params.Op(cuda_input, cuda_output);
//  ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    UnaryInplaceTestFloatSuite, UnaryInplaceTestFloat,
    ::testing::Values(
      // scalar
      UnaryInplaceTestParamsFloat{{}, 
                                  {},
                                  EXP_INPLACE},
      // tensor
      UnaryInplaceTestParamsFloat{{500000}, 
                                  {500000},
                                    EXP_INPLACE},
      UnaryInplaceTestParamsFloat{{600000}, 
                                  {600000},
                                    LOG_INPLACE},
      UnaryInplaceTestParamsFloat{{800000}, 
                                  {800000},
                                    NEG_INPLACE},
      // parallel(size>=2^15)
      UnaryInplaceTestParamsFloat{{330000}, 
                                  {330000},
                                    NEG_INPLACE}
    )
);



// binary Tensor-Tensor
using BinaryTTInplaceOp = std::function<Tensor(Tensor& a, Tensor& b, Tensor& c)>;
auto ADDTT_INPLACE = [](Tensor& a, Tensor& b, Tensor& c)->Tensor{ return add(a, b, c); };
auto SUBTT_INPLACE = [](Tensor& a, Tensor& b, Tensor& c)->Tensor{ return sub(a, b, c); };
auto MULTT_INPLACE = [](Tensor& a, Tensor& b, Tensor& c)->Tensor{ return mul(a, b, c); };
auto DIVTT_INPLACE = [](Tensor& a, Tensor& b, Tensor& c)->Tensor{ return div(a, b, c); };
auto MAXTT_INPLACE = [](Tensor& a, Tensor& b, Tensor& c)->Tensor{ return max(a, b, c); };

template <typename TScalarType>
struct BinaryTTInplaceTestParams {
  std::vector<int64_t> dims_input1;
  std::vector<int64_t> dims_input2;
  std::vector<int64_t> dims_output;
  BinaryTTInplaceOp Op;
};

template <typename TScalarType>
class BinaryTTInplaceTest
    : public ::testing::TestWithParam<BinaryTTInplaceTestParams<TScalarType>> {};

using BinaryTTInplaceTestParamsFloat = BinaryTTInplaceTestParams<float>;
using BinaryTTInplaceTestFloat = BinaryTTInplaceTest<float>;
TEST_P(BinaryTTInplaceTestFloat, RandUniformValueTests) {
  BinaryTTInplaceTestParamsFloat params =
      ::testing::TestWithParam<BinaryTTInplaceTestParamsFloat>::GetParam();
  Tensor cpu_input1 = rand_uniform(params.dims_input1, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_input2 = rand_uniform(params.dims_input2, 1.0, 10.0, device(kCPU).dtype(kFloat));   
  Tensor cpu_output = rand_uniform(params.dims_output, 1.0, 10.0, device(kCPU).dtype(kFloat));  
//  Tensor cuda_output = rand_uniform(params.dims_output, 1.0, 10.0, device(kCUDA).dtype(kFloat));
        struct timeval hice_start, hice_end;
        double hice_time;
        gettimeofday(&hice_start,NULL);

        params.Op(cpu_input1, cpu_input2, cpu_output);
        gettimeofday(&hice_end,NULL);
        hice_time = (hice_end.tv_sec - hice_start.tv_sec) * 1000.0
                    + (hice_end.tv_usec - hice_start.tv_usec) / 1000.0 ;


        std::cout<< /*GREEN <<*/ "\t[ HICE ] " << /*RESET <<*/ hice_time << " ms" << std::endl;
//  Tensor cuda_input1 = cpu_input1.to(kCUDA);
//  Tensor cuda_input2 = cpu_input2.to(kCUDA);
//  params.Op(cuda_input1, cuda_input2, cuda_output);
  // dims compare(cpu, cuda, Truth)
  auto ndim_output_truth = params.dims_output.size();
  EXPECT_EQ(cpu_output.ndim(), ndim_output_truth);
//  EXPECT_EQ(cuda_output.ndim(), ndim_output_truth);
  for (size_t i = 0; i < ndim_output_truth; ++i) {
    EXPECT_EQ(cpu_output.dim(i), params.dims_output[i]);
//    EXPECT_EQ(cuda_output.dim(i), params.dims_output[i]);
  }
  // data compare(cpu, cuda)
  auto size_cpu_output = cpu_output.size();
//  auto size_cuda_output = cuda_output.size();
//  Tensor cuda_output_host = cuda_output.to(kCPU);
//  EXPECT_EQ(size_cpu_output, size_cuda_output);
//  for(int i = 0; i < size_cpu_output; ++i) {
//    EXPECT_FLOAT_EQ(cpu_output.data<float>()[i], cuda_output_host.data<float>()[i]);
//  }
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    BinaryTTInplaceTestFloatSuite, BinaryTTInplaceTestFloat,
    ::testing::Values(
      // none-broadcast
      BinaryTTInplaceTestParamsFloat{{}, 
                                    {}, 
                                    {},
                                    ADDTT_INPLACE},
      BinaryTTInplaceTestParamsFloat{{500000}, 
                                    {500000}, 
                                    {500000},
                                    ADDTT_INPLACE},
      BinaryTTInplaceTestParamsFloat{{600000}, 
                                    {600000}, 
                                    {600000},
                                    SUBTT_INPLACE},
      BinaryTTInplaceTestParamsFloat{{800000}, 
                                    {800000}, 
                                    {800000},
                                    MULTT_INPLACE},
      BinaryTTInplaceTestParamsFloat{{100000}, 
                                    {100000}, 
                                    {100000},
                                    DIVTT_INPLACE},
      BinaryTTInplaceTestParamsFloat{{300000}, 
                                    {300000}, 
                                    {300000},
                                    MAXTT_INPLACE},
      // broadcast
      BinaryTTInplaceTestParamsFloat{{}, 
                                    {5, 6, 7}, 
                                    {5, 6, 7},
                                    ADDTT_INPLACE},
      BinaryTTInplaceTestParamsFloat{{5, 6, 7}, 
                                    {}, 
                                    {5, 6, 7},
                                    ADDTT_INPLACE},
      BinaryTTInplaceTestParamsFloat{{1, 6, 7}, 
                                    {5, 6, 7}, 
                                    {5, 6, 7},
                                    ADDTT_INPLACE},
      BinaryTTInplaceTestParamsFloat{{1, 1, 1}, 
                                    {2, 3, 5}, 
                                    {2, 3, 5},
                                    SUBTT_INPLACE},
      BinaryTTInplaceTestParamsFloat{{1, 6, 1, 4}, 
                                    {2, 1, 3, 1}, 
                                    {2, 6, 3, 4},
                                    MULTT_INPLACE},
      BinaryTTInplaceTestParamsFloat{         {2}, 
                                    {2, 3, 1, 2}, 
                                    {2, 3, 1, 2},
                                    DIVTT_INPLACE},
      // broadcast && parallel
      BinaryTTInplaceTestParamsFloat{{1, 60, 70}, 
                                    {8, 60, 70}, 
                                    {8, 60, 70},
                                    MAXTT_INPLACE}
      )
);




// binary Tensor-Scalar(numerical)
using BinaryTSInplaceOp = std::function<Tensor(Tensor& a, Scalar b, Tensor& c)>;
auto ADDTS_INPLACE = [](Tensor& a, Scalar b, Tensor& c)->Tensor{ return add(a, b, c); };
auto SUBTS_INPLACE = [](Tensor& a, Scalar b, Tensor& c)->Tensor{ return sub(a, b, c); };
auto MULTS_INPLACE = [](Tensor& a, Scalar b, Tensor& c)->Tensor{ return mul(a, b, c); };
auto DIVTS_INPLACE = [](Tensor& a, Scalar b, Tensor& c)->Tensor{ return div(a, b, c); };
auto MAXTS_INPLACE = [](Tensor& a, Scalar b, Tensor& c)->Tensor{ return max(a, b, c); };

template <typename TScalarType>
struct BinaryTSInplaceTestParams {
  std::vector<int64_t> dims_input1;
  std::vector<int64_t> dims_output;
  BinaryTSInplaceOp Op;
};

template <typename TScalarType>
class BinaryTSInplaceTest
    : public ::testing::TestWithParam<BinaryTSInplaceTestParams<TScalarType>> {};

using BinaryTSInplaceTestParamsFloat = BinaryTSInplaceTestParams<float>;
using BinaryTSInplaceTestFloat = BinaryTSInplaceTest<float>;
TEST_P(BinaryTSInplaceTestFloat, RandUniformValueTests) {
  BinaryTSInplaceTestParamsFloat params =
      ::testing::TestWithParam<BinaryTSInplaceTestParamsFloat>::GetParam();
  Tensor cpu_input1 = rand_uniform(params.dims_input1, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_output = rand_uniform(params.dims_output, 1.0, 10.0, device(kCPU).dtype(kFloat));
//  Tensor cuda_input1 = cpu_input1.to(kCUDA);
//  Tensor cuda_output = rand_uniform(params.dims_output, 1.0, 10.0, device(kCUDA).dtype(kFloat));
  float scalar = cpu_input1.data<float>()[0];
        struct timeval hice_start, hice_end;
        double hice_time;
        gettimeofday(&hice_start,NULL);

        params.Op(cpu_input1, scalar, cpu_output);
        gettimeofday(&hice_end,NULL);
        hice_time = (hice_end.tv_sec - hice_start.tv_sec) * 1000.0
                    + (hice_end.tv_usec - hice_start.tv_usec) / 1000.0 ;


        std::cout<< /*GREEN <<*/ "\t[ HICE ] " << /*RESET <<*/ hice_time << " ms" << std::endl;
//  params.Op(cuda_input1, scalar, cuda_output);
//  ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    BinaryTSInplaceTestFloatSuite, BinaryTSInplaceTestFloat,
    ::testing::Values(
      // broadcast
      BinaryTSInplaceTestParamsFloat{{5}, 
                                    {5},
                                    ADDTS_INPLACE},
      BinaryTSInplaceTestParamsFloat{{2, 3, 1, 2}, 
                                    {2, 3, 1, 2},
                                    SUBTS_INPLACE},
      BinaryTSInplaceTestParamsFloat{{2, 6, 3, 4}, 
                                    {2, 6, 3, 4},
                                    MULTS_INPLACE},
      BinaryTSInplaceTestParamsFloat{{2, 3, 5}, 
                                    {2, 3, 5},
                                    DIVTS_INPLACE},
      BinaryTSInplaceTestParamsFloat{{5, 6, 7}, 
                                    {5, 6, 7},
                                    MAXTS_INPLACE},
      // broadcast && parallel
      BinaryTSInplaceTestParamsFloat{{8, 60, 70}, 
                                    {8, 60, 70},
                                    MAXTS_INPLACE}
    )
);




// bianry Scalar-Tensor
using BinarySTInplaceOp = std::function<Tensor(Scalar a, Tensor& b, Tensor& c)>;
auto ADDST_INPLACE = [](Scalar a, Tensor& b, Tensor& c)->Tensor{ return add(a, b, c); };
auto SUBST_INPLACE = [](Scalar a, Tensor& b, Tensor& c)->Tensor{ return sub(a, b, c); };
auto MULST_INPLACE = [](Scalar a, Tensor& b, Tensor& c)->Tensor{ return mul(a, b, c); };
auto DIVST_INPLACE = [](Scalar a, Tensor& b, Tensor& c)->Tensor{ return div(a, b, c); };
auto MAXST_INPLACE = [](Scalar a, Tensor& b, Tensor& c)->Tensor{ return max(a, b, c); };

template <typename TScalarType>
struct BinarySTInplaceTestParams {
  std::vector<int64_t> dims_input2;
  std::vector<int64_t> dims_output;
  BinarySTInplaceOp Op;
};

template <typename TScalarType>
class BinarySTInplaceTest
    : public ::testing::TestWithParam<BinarySTInplaceTestParams<TScalarType>> {};

using BinarySTInplaceTestParamsFloat = BinarySTInplaceTestParams<float>;
using BinarySTInplaceTestFloat = BinarySTInplaceTest<float>;
TEST_P(BinarySTInplaceTestFloat, RandUniformValueTests) {
  BinarySTInplaceTestParamsFloat params =
      ::testing::TestWithParam<BinarySTInplaceTestParamsFloat>::GetParam();
  Tensor cpu_input2 = rand_uniform(params.dims_input2, 1.0, 10.0, device(kCPU).dtype(kFloat));
//  Tensor cuda_input2 = cpu_input2.to(kCUDA);
  Tensor cpu_output = rand_uniform(params.dims_output, 1.0, 10.0, device(kCPU).dtype(kFloat));  
//  Tensor cuda_output = rand_uniform(params.dims_output, 1.0, 10.0, device(kCUDA).dtype(kFloat));
  float scalar = cpu_input2.data<float>()[0];
        struct timeval hice_start, hice_end;
        double hice_time;
        gettimeofday(&hice_start,NULL);

        params.Op(scalar, cpu_input2, cpu_output);
        gettimeofday(&hice_end,NULL);
        hice_time = (hice_end.tv_sec - hice_start.tv_sec) * 1000.0
                    + (hice_end.tv_usec - hice_start.tv_usec) / 1000.0 ;


        std::cout<< /*GREEN <<*/ "\t[ HICE ] " << /*RESET <<*/ hice_time << " ms" << std::endl;
//  params.Op(scalar, cuda_input2, cuda_output);
//  ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    BinarySTInplaceTestFloatSuite, BinarySTInplaceTestFloat,
    ::testing::Values(
      // broadcast
      BinarySTInplaceTestParamsFloat{{5}, 
                                    {5},
                                    ADDST_INPLACE},
      BinarySTInplaceTestParamsFloat{{2, 3, 1, 2}, 
                                    {2, 3, 1, 2},
                                    SUBST_INPLACE},
      BinarySTInplaceTestParamsFloat{{2, 3, 1, 2}, 
                                    {2, 3, 1, 2},
                                    SUBST_INPLACE},
      BinarySTInplaceTestParamsFloat{{2, 6, 3, 4}, 
                                    {2, 6, 3, 4},
                                    MULST_INPLACE},
      BinarySTInplaceTestParamsFloat{{2, 3, 5}, 
                                    {2, 3, 5},
                                    DIVST_INPLACE},
      BinarySTInplaceTestParamsFloat{{5, 6, 7}, 
                                    {5, 6, 7},
                                    MAXST_INPLACE},
      // broadcast && parallel
      BinarySTInplaceTestParamsFloat{{8, 60, 70}, 
                                    {8, 60, 70},
                                    MAXST_INPLACE}
    )
);

#endif
} // namespace hice
 
