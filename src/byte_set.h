#include <parallel_hashmap/phmap_utils.h>
//#include <stdio.h>

template <int T> struct byte_set {
  char data_[T];
  byte_set() {};
  byte_set(void* d) { memcpy(data_, d, T); };
  bool operator==(const byte_set &other) const {
    return memcmp(data_, other.data_, T) == 0;
  }
  friend std::size_t hash_value(const byte_set<T> &b) {
    return std::hash<std::string>{}(std::string(b.data_, T));
  };
//  void print() {
//    for(int i=0; i<T; i++) fprintf(stdout, "%02hhx ", data_[i]);
//    fprintf(stdout, "\n");
// };
};


// using T = typename std::conditional_t<J == 1, std::int8_t,
//   std::conditional_t<J == 2, std::int16_t,
//     std::conditional_t<J == 4, std::int32_t,
//       std::conditional_t<J == 8, std::int64_t, std::int64_t>>>>;


template<> struct byte_set<1> {
  using I = std::int8_t;
  I data_;
  byte_set() {};
  byte_set(void* d) { auto p = (I* )d; data_ = p[0]; };
  bool operator==(const byte_set &other) const {
    return data_ == other.data_;
  }
  friend std::size_t hash_value(const byte_set<1> &b) {
    return std::hash<I>{}(b.data_);
  }
};


template<> struct byte_set<2> {
  using I = std::int16_t;
  I data_;
  byte_set() {};
  byte_set(void* d) { auto p = (I* )d; data_ = p[0]; };
  bool operator==(const byte_set &other) const {
    return data_ == other.data_;
  }
  friend std::size_t hash_value(const byte_set<2> &b) {
    return std::hash<I>{}(b.data_);
  }
};


template<> struct byte_set<4> {
  using I = std::int32_t;
  I data_;
  byte_set() {};
  byte_set(void* d) { auto p = (I* )d; data_ = p[0]; };
  bool operator==(const byte_set &other) const {
    return data_ == other.data_;
  }
  friend std::size_t hash_value(const byte_set<4> &b) {
    return std::hash<I>{}(b.data_);
  }
};


template<> struct byte_set<8> {
  using I = std::int64_t;
  I data_;
  byte_set() {};
  byte_set(void* d) { auto p = (I* )d; data_ = p[0]; };
  bool operator==(const byte_set &other) const {
    return data_ == other.data_;
  }
  friend std::size_t hash_value(const byte_set<8> &b) {
    return std::hash<I>{}(b.data_);
  }
};
