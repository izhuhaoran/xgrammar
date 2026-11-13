/*!
 * Copyright (c) 2026 by Contributors
 * \file xgrammar/object_key_context.h
 * \brief Persistent per-derivation key sets for any-order objects.
 */
#ifndef XGRAMMAR_OBJECT_KEY_CONTEXT_H_
#define XGRAMMAR_OBJECT_KEY_CONTEXT_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "support/encoding.h"
#include "support/utils.h"

namespace xgrammar {

/*!
 * Parser states carry a small integer rather than copying a set on each byte.
 * Sets are persistent binary tries; adding a key allocates at most 32 small nodes,
 * independent of the number of keys already present. A negative leaf stores a
 * single key, avoiding the remaining unary path. Interning also keeps equal
 * Earley derivations equal. Speculative rows roll back their allocations.
 */
class ObjectKeyContext {
  enum ScopeField { kOuter, kRule, kStart, kSeen, kPrefix, kCodec, kName, kTail };
  using Scope = std::array<int32_t, 8>;
  template <size_t N>
  class Table {
    using Value = std::array<int32_t, N>;
    struct Hash {
      size_t operator()(const Value& value) const {
        size_t result = 0;
        for (int32_t part : value) result = HashCombine(result, part);
        return result;
      }
    };
    std::vector<Value> values_;
    std::unordered_map<Value, int32_t, Hash> ids_;

   public:
    int32_t Intern(const Value& value) {
      if (!values_.empty() && values_.back() == value) return values_.size() - 1;
      auto [it, inserted] = ids_.try_emplace(value, values_.size());
      if (inserted) values_.push_back(value);
      return it->second;
    }
    const Value& operator[](int32_t id) const { return values_[id]; }
    size_t Size() const { return values_.size(); }
    void Resize(size_t size) {
      while (values_.size() > size) {
        ids_.erase(values_.back());
        values_.pop_back();
      }
    }
  };

  // Keep the latest seen key inline. A speculative raw-name completion can
  // then register its name without copying a trie path for every possible prefix.
  // kName is the canonical prefix of an unescaped JSON key; escaped keys retain
  // the existing decoder. A parser context is [current scope, rule-entry scope].
  // The entry identity
  // prevents a completion from returning through an unrelated Earley branch.
  Table<2> sets_;
  Table<8> scopes_;
  Table<2> states_;
  Table<2> prefixes_;

  static int32_t StateId(int32_t state) { return state < 0 ? ~state : state; }
  int32_t State(int32_t current, int32_t entry) {
    int32_t id = states_.Intern({current, entry});
    // Only key bytes need prefix tracking. Values take the ordinary positive-ID path.
    return scopes_[current][kPrefix] >= 0 ? ~id : id;
  }

 public:
  using Snapshot = std::array<size_t, 4>;
  static constexpr int32_t kRejected = std::numeric_limits<int32_t>::min();

 private:
  struct Checkpoint {
    int32_t row;
    Snapshot sizes;
  };
  std::vector<Checkpoint> checkpoints_;
  int32_t transactions_ = 0;
  // Prediction visits several alternatives with the same preceding key.
  std::array<int32_t, 3> last_insert_{-1, -1, -1};

  bool Contains(int32_t root, uint32_t key) const {
    for (int bit = 0; root > 0; ++bit) root = sets_[root][(key >> bit) & 1];
    return root == ~static_cast<int32_t>(key);
  }
  // Low bits separate the usual small name IDs. A singleton subtree compares
  // the whole key at its leaf instead of walking the remaining bits.
  int32_t Insert(int32_t root, uint32_t key, int bit = 0) {
    const int32_t leaf = ~static_cast<int32_t>(key);
    if (root == 0 || root == leaf) return leaf;
    auto children = root > 0 ? sets_[root] : std::array<int32_t, 2>{0, 0};
    if (root < 0) children[(static_cast<uint32_t>(~root) >> bit) & 1] = root;
    const int branch = (key >> bit) & 1;
    children[branch] = Insert(children[branch], key, bit + 1);
    return sets_.Intern(children);
  }
  bool HasSeen(const Scope& scope, int32_t key) const {
    return scope[kTail] == key || Contains(scope[kSeen], key);
  }
  void MaterializeTail(Scope* scope) {
    if ((*scope)[kTail] >= 0) {
      const int32_t root = (*scope)[kSeen], key = (*scope)[kTail];
      if (last_insert_[0] != root || last_insert_[1] != key) {
        last_insert_ = {root, key, Insert(root, key)};
      }
      (*scope)[kSeen] = last_insert_[2];
      (*scope)[kTail] = -1;
    }
  }
  int32_t RememberKey(const std::array<int32_t, 2>& pair, Scope scope, int32_t key) {
    if (HasSeen(scope, key)) return kRejected;
    MaterializeTail(&scope);
    scope[kTail] = key;
    return State(scopes_.Intern(scope), pair[1]);
  }

 public:
  ObjectKeyContext() {
    sets_.Intern({0, 0});
    scopes_.Intern({0, -1, -1, 0, -1, 0, 0, -1});
    states_.Intern({0, 0});
    prefixes_.Intern({-1, 0});
    checkpoints_.push_back({0, Save()});
  }
  Snapshot Save() const { return {sets_.Size(), scopes_.Size(), states_.Size(), prefixes_.Size()}; }
  void Restore(const Snapshot& snapshot) {
    last_insert_ = {-1, -1, -1};
    states_.Resize(snapshot[2]);
    scopes_.Resize(snapshot[1]);
    sets_.Resize(snapshot[0]);
    prefixes_.Resize(snapshot[3]);
  }
  void Mark(int32_t row) {
    const auto sizes = Save();
    if (checkpoints_.back().sizes == sizes) return;
    if (checkpoints_.back().row == row)
      checkpoints_.back().sizes = sizes;
    else
      checkpoints_.push_back({row, sizes});
  }
  void Rollback(int32_t row) {
    while (checkpoints_.size() > 1 && checkpoints_.back().row > row) checkpoints_.pop_back();
    if (transactions_ == 0) Restore(checkpoints_.back().sizes);
  }
  struct Transaction {
    Snapshot sizes;
    size_t checkpoints;
    Checkpoint last;
  };
  Transaction BeginTransaction() {
    ++transactions_;
    return {Save(), checkpoints_.size(), checkpoints_.back()};
  }
  void EndTransaction(const Transaction& transaction, bool accepted, int32_t row) {
    --transactions_;
    if (accepted)
      Mark(row);
    else {
      Restore(transaction.sizes);
      checkpoints_.resize(transaction.checkpoints);
      checkpoints_.back() = transaction.last;
    }
  }
  int32_t InternKey(const std::string& key) {
    int32_t prefix = 0;
    for (uint8_t byte : key) prefix = prefixes_.Intern({prefix, byte});
    return prefix;
  }

  int32_t Enter(int32_t parent, int32_t rule, int32_t start, bool object, int32_t key) {
    const auto parent_pair = states_[StateId(parent)];
    const int32_t entry = parent_pair[0];
    if (!object && key == -1 && entry == parent_pair[1]) return parent;
    int32_t current = entry;
    if (object) {
      current = scopes_.Intern({entry, rule, start, 0, -1, 0, 0, -1});
    } else if (key >= 0 && current != 0) {
      auto scope = scopes_[current];
      if (HasSeen(scope, key)) return kRejected;
      // Check before matching the property, but register only the selected
      // property at completion. Prediction must not allocate a set per alternative.
      if (scope[kTail] >= 0) {
        MaterializeTail(&scope);
        current = scopes_.Intern(scope);
      }
      if (current == entry && entry == parent_pair[1]) return parent;
    } else if (key <= -3 && current != 0) {
      auto scope = scopes_[current];
      MaterializeTail(&scope);
      scope[kPrefix] = 0;
      scope[kCodec] = -key - 2;  // 1: unescaped JSON, 2: raw, 3: fallback, 4: JSON closed.
      scope[kName] = 0;
      current = scopes_.Intern(scope);
    }
    return State(current, entry);
  }
  int32_t AdvanceKeyByte(int32_t state, uint8_t byte) {
    const auto pair = states_[StateId(state)];
    auto scope = scopes_[pair[0]];
    if (scope[kCodec] == 1) {
      if (scope[kPrefix] == 0) {
        if (byte != '"') scope[kCodec] = 3;
      } else if (byte == '\\') {
        scope[kCodec] = 3;
      } else if (byte == '"') {
        scope[kCodec] = 4;
      } else {
        scope[kName] = prefixes_.Intern({scope[kName], byte});
      }
    } else if (scope[kCodec] == 4) {
      // A quote followed by more bytes is not the final quote. Preserve the
      // original decoder's behavior for custom annotated grammars too.
      scope[kCodec] = 3;
    }
    scope[kPrefix] = prefixes_.Intern({scope[kPrefix], byte});
    return State(scopes_.Intern(scope), pair[1]);
  }
  int32_t CompleteDeclaredKey(int32_t state, int32_t key) {
    const auto pair = states_[StateId(state)];
    return pair[0] == 0 ? state : RememberKey(pair, scopes_[pair[0]], key);
  }
  int32_t CompleteKey(int32_t state) {
    const auto pair = states_[StateId(state)];
    if (pair[0] == 0) return state;
    auto scope = scopes_[pair[0]];
    // Raw keys already have a canonical prefix ID. Retaining a full string for
    // every possible raw-name prefix would require quadratic memory on long keys.
    int32_t name = scope[kCodec] == 4 ? scope[kName] : scope[kPrefix];
    if (scope[kCodec] == 1) return kRejected;
    if (scope[kCodec] == 3) {
      std::string key;
      for (int32_t prefix = scope[kPrefix]; prefix > 0; prefix = prefixes_[prefix][0]) {
        key.push_back(static_cast<char>(prefixes_[prefix][1]));
      }
      std::reverse(key.begin(), key.end());
      if (key.size() < 2 || key.front() != '"' || key.back() != '"') return kRejected;
      std::string decoded;
      for (size_t i = 1; i + 1 < key.size();) {
        if (key[i] != '\\') {
          decoded.push_back(key[i++]);
          continue;
        }
        auto [codepoint, length] = ParseNextEscaped(key.c_str() + i);
        if (codepoint < 0 || i + length >= key.size()) return kRejected;
        i += length;
        if (codepoint >= 0xd800 && codepoint <= 0xdbff && i + 6 < key.size() && key[i] == '\\' &&
            key[i + 1] == 'u') {
          const auto [low, low_length] = ParseNextEscaped(key.c_str() + i);
          if (low >= 0xdc00 && low <= 0xdfff) {
            codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
            i += low_length;
          }
        }
        // Preserve lone surrogate escapes accepted by the existing string grammar.
        // This performs JSON unescaping, not Unicode normalization or extra validation.
        decoded += CharToUTF8(codepoint);
      }
      name = InternKey(decoded);
    }
    scope[kPrefix] = -1;
    scope[kCodec] = 0;
    scope[kName] = 0;
    return RememberKey(pair, scope, name);
  }
  int32_t EntryScope(int32_t state) const { return states_[StateId(state)][1]; }
  // The caller must test compatibility before collecting captures or propagating
  // a rule completion. A different entry set denotes a different derivation.
  bool CanReturn(int32_t child, int32_t parent) const {
    return states_[StateId(child)][1] == states_[StateId(parent)][0];
  }
  int32_t Return(int32_t child, int32_t parent, bool object) {
    int32_t current = states_[StateId(child)][0];
    if (object && current != 0) current = scopes_[current][kOuter];
    if (current == states_[StateId(parent)][0]) return parent;
    return State(current, states_[StateId(parent)][1]);
  }
  int32_t RebaseRow(int32_t state, int32_t from, int32_t to) {
    if (state == 0 || from == to) return state;
    auto rebase_scope = [&](int32_t scope) {
      // Only the atomic closure's newly entered scopes use this row. Their
      // outer chain ends at an older scope, which keeps its existing identity.
      std::vector<Scope> chain;
      while (scope != 0 && scopes_[scope][kStart] == from) {
        chain.push_back(scopes_[scope]);
        scope = chain.back()[kOuter];
      }
      for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        auto value = *it;
        value[kStart] = to;
        value[kOuter] = scope;
        scope = scopes_.Intern(value);
      }
      return scope;
    };
    const auto pair = states_[StateId(state)];
    const int32_t current = rebase_scope(pair[0]), entry = rebase_scope(pair[1]);
    return current == pair[0] && entry == pair[1] ? state : State(current, entry);
  }
};

}  // namespace xgrammar
#endif  // XGRAMMAR_OBJECT_KEY_CONTEXT_H_
