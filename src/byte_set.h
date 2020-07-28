#include <parallel_hashmap/phmap_utils.h>

template <int T> struct byte_set {
  char data_[T];
  byte_set() {};
  byte_set(void* d) { memcpy(data_, d, T); };
  bool operator==(const byte_set &other) const {
    return memcmp(data_, other.data_, T) == 0;
  }
  friend std::size_t hash_value(const byte_set<T> &b) {
    return std::hash<std::string>{}(std::string(b.data_, T));
  }
};


// using T = typename std::conditional_t<J == 1, std::int8_t,
//   std::conditional_t<J == 2, std::int16_t,
//     std::conditional_t<J == 4, std::int32_t,
//       std::conditional_t<J == 8, std::int64_t, std::int64_t>>>>;

// template<> struct byte_set<1>  {
//   std::uint8_t data_;
//   bool operator==(const byte_set &other) const {
//     return data_ == other.data_;
//   }
//   friend std::size_t hash_value(const byte_set<1> &b) {
//     return std::hash<std::uint8_t>{}(b.data_);
//   }
// };
//
//
// template<> struct byte_set<2> {
//   std::uint16_t data_;
//   bool operator==(const byte_set &other) const {
//     return data_ == other.data_;
//   }
//   friend std::size_t hash_value(const byte_set<2> &b) {
//     return std::hash<std::uint16_t>{}(b.data_);
//   }
// };
//
//
// template<> struct byte_set<4> {
//   std::uint32_t data_;
//   bool operator==(const byte_set &other) const {
//     return data_ == other.data_;
//   }
//   friend std::size_t hash_value(const byte_set<4> &b) {
//     return std::hash<std::uint32_t>{}(b.data_);
//   }
// };
//
//
// template<> struct byte_set<8>  {
//   std::uint64_t data_;
//   bool operator==(const byte_set &other) const {
//     return data_ == other.data_;
//   }
//   friend std::size_t hash_value(const byte_set<8> &b) {
//     return std::hash<std::uint64_t>{}(b.data_);
//   }
// };
