#pragma once

template <typename T, size_t n> 
struct AtomicAddIntegerImpl;

template <typename T> 
struct AtomicAddIntegerImpl<T, 1> {
  inline __device__ void operator()(T *address, T val) {
    uint32_t *address_as_ui = (uint32_t *)(address - ((size_t)address & 3));
    uint32_t old = *address_as_ui;
    uint32_t shift = (((size_t)address & 3) * 8);
    uint32_t sum;
    uint32_t assumed;

    do {
      assumed = old;
      sum = val + T((old >> shift) & 0xff);
      old = (old & ~(0x000000ff << shift)) | (sum << shift);
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
  }
};

template <typename T> 
struct AtomicAddIntegerImpl<T, 2> {
  inline __device__ void operator()(T *address, T val) {
    uint32_t *address_as_ui =
        (uint32_t *)((char *)address - ((size_t)address & 2));
    uint32_t old = *address_as_ui;
    uint32_t sum;
    uint32_t newval;
    uint32_t assumed;

    do {
      assumed = old;
      sum = val + (size_t)address & 2 ? T(old >> 16) : T(old & 0xffff);
      newval = (size_t)address & 2 ? (old & 0xffff) | (sum << 16)
                                   : (old & 0xffff0000) | sum;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template <typename T> 
struct AtomicAddIntegerImpl<T, 4> {
  inline __device__ void operator()(T *address, T val) {
    uint32_t *address_as_ui = (uint32_t *)(address);
    uint32_t old = *address_as_ui;
    uint32_t newval;
    uint32_t assumed;

    do {
      assumed = old;
      newval = val + (T)old;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template <typename T> 
struct AtomicAddIntegerImpl<T, 8> {
  inline __device__ void operator()(T *address, T val) {
    unsigned long long *address_as_ui = (unsigned long long *)(address);
    unsigned long long old = *address_as_ui;
    unsigned long long newval;
    unsigned long long assumed;

    do {
      assumed = old;
      newval = val + (T)old;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

static inline __device__ void atomicAdd(uint8_t *address, uint8_t val) {
  AtomicAddIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val);
}

static inline __device__ void atomicAdd(int8_t *address, int8_t val) {
  AtomicAddIntegerImpl<int8_t, sizeof(int8_t)>()(address, val);
}

static inline __device__ void atomicAdd(int16_t *address, int16_t val) {
  AtomicAddIntegerImpl<int16_t, sizeof(int16_t)>()(address, val);
}

static inline __device__ void atomicAdd(int64_t *address, int64_t val) {
  AtomicAddIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
}