/* Implementation of C++ version of "enumerate()" */
/* Code by Minjang Kim. (http://minjang.github.io/2016/01/07/python-enumerate-modern-c++/) */
#ifndef __NBA_CORE_ENUM_HH__
#define __NBA_CORE_ENUM_HH__

#include <cstddef>
#include <vector>
#include <typeinfo>
#include <utility>

namespace nba {

template<typename C>
class enumerate_iterator {
  using IndexType = size_t;
  using IterType = decltype(std::begin(std::declval<C&>()));
  using ElemType = decltype(*std::begin(std::declval<C&>()));
  using PairType = std::pair<IndexType, ElemType>;

  IterType it_;
  size_t index_;

public:
  enumerate_iterator(IterType it, size_t index) :
    it_{it}, index_{index} {}

  enumerate_iterator& operator++() {
    ++it_;
    ++index_;
    return *this;
  }

  bool operator!=(const enumerate_iterator& rhs) {
    return it_ != rhs.it_;
  }

  PairType operator*() {
    return {index_, *it_};
  }
};

template<typename C>
class enumerate_impl {
  C container_;
  const size_t start_;

public:
  enumerate_impl(C&& container, size_t start) :
    container_(std::forward<C>(container)), start_{start} {
    // In G++ 4.8, above forwarded rvalue ref to container should use
    // parenthesis instead of braces (initializer list).
  }

  enumerate_iterator<C> begin() {
    return {std::begin(container_), start_};
  }

  enumerate_iterator<C> end() {
    return {std::end(container_), 0 /* not important */};
  }
};

/* Forwarding reference version: supports both lvalue/rvalue */
template<typename C>
enumerate_impl<C> enumerate(C&& container, size_t start = 0) {
  return {std::forward<C>(container), start};
}

/* Initializer list version */
template<typename T>
enumerate_impl<std::vector<T>> enumerate(std::initializer_list<T> list, size_t start = 0) {
  return {std::vector<T>(list), start};
}

} /* endns(nba) */

#endif /* __NBA_CORE_ENUM_HH__ */

// vim: ts=8 sts=2 sw=2 et
