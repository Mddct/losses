#include "ctc_decoder.h"

#include <cinttypes>
#include <cstdio>
#include <iostream>
#include <vector>

template <typename T, typename Cmp>
int64_t LevenshteinDistance(const std::vector<T> &s, const int64_t s_seq_size,
                            const std::vector<T> &t, const int64_t t_seq_size,
                            const Cmp &cmp) {
  int64_t s_size = s_seq_size;
  int64_t t_size = t_seq_size;
  if (s.size() < s_seq_size) {
    s_size = s.size();
  }

  if (t.size() < t_seq_size) {
    t_size = t.size();
  }

  if (t_size > s_size)
    return LevenshteinDistance(t, t_size, s, s_size, cmp);

  const T *s_data = s.data();
  const T *t_data = t.data();

  if (t_size == 0)
    return s_size;
  if (s == t)
    return 0;

  // Create work vector
  std::vector<T> scratch_holder(t_size);

  int64_t *scratch = scratch_holder.data();

  // Special case for i = 0: Distance between empty string and string
  // of length j is just j.
  for (size_t j = 1; j < t_size; ++j)
    scratch[j - 1] = j;

  for (size_t i = 1; i <= s_size; ++i) {
    // Invariant: scratch[j - 1] equals cost(i - 1, j).
    int substitution_base_cost = i - 1;
    int insertion_cost = i + 1;
    for (size_t j = 1; j <= t_size; ++j) {
      // Invariants:
      //  scratch[k - 1] = cost(i, k)  for 0 < k < j.
      //  scratch[k - 1] = cost(i - 1, k)  for j <= k <= t_size.
      //  substitution_base_cost = cost(i - 1, j - 1)
      //  insertion_cost = cost(i, j - 1)
      const int replacement_cost = cmp(s_data[i - 1], t_data[j - 1]) ? 0 : 1;
      const int substitution_cost = substitution_base_cost + replacement_cost;
      const int deletion_cost = scratch[j - 1] + 1;

      // Select the cheapest edit.
      const int cheapest = // = cost(i, j)
          std::min(deletion_cost, std::min(insertion_cost, substitution_cost));

      // Restore invariant for the next iteration of the loop.
      substitution_base_cost = scratch[j - 1]; // = cost(i - 1, j)
      scratch[j - 1] = cheapest;               // = cost(i, j)
      insertion_cost = cheapest + 1;           // = cost(i, j) + 1
    }
  }
  return scratch[t_size - 1];
}

// sdata: shape [bs, s_max_seq_len]
// tdata: shape [bs, t_max_seq_len]
std::vector<int64_t> EditDistance(uintptr_t sdata, int s_max_seq_len,
                                  uintptr_t s_sequence_length, uintptr_t tdata,
                                  int t_max_seq_len,
                                  uintptr_t t_sequence_length, int bs) {
  if (bs <= 0) {
    return std::vector<int64_t>{};
  }

  auto *s = reinterpret_cast<int64_t *>(sdata);
  auto *s_seq_lens = reinterpret_cast<int64_t *>(s_sequence_length);
  auto *t = reinterpret_cast<int64_t *>(tdata);
  auto *t_seq_lens = reinterpret_cast<int64_t *>(t_sequence_length);

  auto cmp = std::equal_to<int64_t>();

  std::vector<int64_t> distances;
  for (int i = 0; i < bs; i++) {
    std::vector<int64_t> source(s + i * s_max_seq_len,
                                s + i * s_max_seq_len + s_seq_lens[i]);
    std::vector<int64_t> dst(t + i * t_max_seq_len,
                             t + i * t_max_seq_len + t_seq_lens[i]);
    // int64_t LevenshteinDistance(
    //     const std::vector<T> &s, const int64_t s_seq_size,
    //     const std::vector<T> &t, const int64_t t_seq_size, const Cmp &cmp) {
    auto distance =
        LevenshteinDistance(source, source.size(), dst, dst.size(), cmp);
    distances.push_back(distance);
  }
  return std::move(distances);
}

// int main() {
//   std::vector<int64_t> t{1, 2, 3, 4, 5, 6, 6};
//   std::vector<int64_t> s{1, 2};
//   auto cmp = std::equal_to<int64_t>();

//   std::cout << LevenshteinDistance(s, s.size(), t, t.size(), cmp) <<
//   std::endl; std::cout << LevenshteinDistance(s, 2, t, 1, cmp) << std::endl;
// }
