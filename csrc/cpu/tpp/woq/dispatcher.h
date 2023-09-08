#ifndef _TLA_DISPATCHER_H_
#define _TLA_DISPATCHER_H_

#include <cstddef>
#include <utility>

template <typename... Args>
void failing_fallback(Args... args) {
  TLA_ASSERT(false, "should not reach here");
}

// TODO(jgong5): use switch-case to speed up range and enumerate dispatchers

// dispatch a range of integers to a lambda function
template <typename IntegralType, IntegralType start, IntegralType end>
struct range_dispatcher {
  template <typename Lambda1, typename Lambda2, typename... Args>
  inline static void call(
      IntegralType i,
      const Lambda1& function,
      const Lambda2& fallback,
      Args... args) {
    static_assert(start < end, "Start must be less than end");
    if (i == start)
      function(
          std::integral_constant<IntegralType, start>{},
          std::forward<Args>(args)...);
    else
      range_dispatcher<IntegralType, start + 1, end>::call(
          i, function, fallback, std::forward<Args>(args)...);
  }
};

template <typename IntegralType, IntegralType start>
struct range_dispatcher<IntegralType, start, start> {
  template <typename Lambda1, typename Lambda2, typename... Args>
  inline static void call(
      IntegralType i,
      const Lambda1& function,
      const Lambda2& fallback,
      Args... args) {
    if (i == start) {
      function(
          std::integral_constant<IntegralType, start>{},
          std::forward<Args>(args)...);
    } else {
      fallback(i, std::forward<Args>(args)...);
    }
  }
};

template <
    typename IntegralType,
    int n,
    IntegralType First,
    IntegralType... Rest>
struct enumerate_dispatcher_helper {
  template <typename Lambda1, typename Lambda2, typename... Args>
  inline static void call(
      IntegralType i,
      const std::function<bool(IntegralType, IntegralType)>& comparator,
      const Lambda1& function,
      const Lambda2& fallback,
      Args... args) {
    if (comparator(i, First))
      function(
          std::integral_constant<IntegralType, First>{},
          std::forward<Args>(args)...);
    else
      enumerate_dispatcher_helper<IntegralType, n - 1, Rest...>::call(
          i, comparator, function, fallback, std::forward<Args>(args)...);
  }
};

template <typename IntegralType, IntegralType First>
struct enumerate_dispatcher_helper<IntegralType, 0, First> {
  template <typename Lambda1, typename Lambda2, typename... Args>
  inline static void call(
      IntegralType i,
      const std::function<bool(IntegralType, IntegralType)>& comparator,
      const Lambda1& function,
      const Lambda2& fallback,
      Args... args) {
    if (comparator(i, First))
      function(
          std::integral_constant<IntegralType, First>{},
          std::forward<Args>(args)...);
    else
      fallback(i, std::forward<Args>(args)...);
  }
};

// dispatch a list of integers to a lambda function
template <typename IntegralType, IntegralType... ints>
struct enumerate_dispatcher {
  template <typename Lambda1, typename Lambda2, typename... Args>
  inline static void call(
      IntegralType i,
      const Lambda1& function,
      const Lambda2& fallback,
      Args... args) {
    enumerate_dispatcher_helper<IntegralType, sizeof...(ints) - 1, ints...>::
        call(
            i,
            [&](IntegralType a, IntegralType b) { return a == b; },
            function,
            fallback,
            std::forward<Args>(args)...);
  }
};

// dispatch a list of integers to a lambda function based on divisibility
template <typename IntegralType, IntegralType... ints>
struct divisible_dispatcher {
  template <typename Lambda1, typename Lambda2, typename... Args>
  inline static void call(
      IntegralType i,
      const Lambda1& function,
      const Lambda2& fallback,
      Args... args) {
    enumerate_dispatcher_helper<IntegralType, sizeof...(ints) - 1, ints...>::
        call(
            i,
            [&](IntegralType a, IntegralType b) { return a % b == 0; },
            function,
            fallback,
            std::forward<Args>(args)...);
  }
};

// dispatch a list of integers to a lambda function based on given condition
template <typename IntegralType, IntegralType... ints>
struct conditional_enumerate_dispatcher {
  template <typename Lambda1, typename Lambda2, typename... Args>
  inline static void call(
      IntegralType i,
      const std::function<bool(IntegralType, IntegralType)>& comparator,
      const Lambda1& function,
      const Lambda2& fallback,
      Args... args) {
    enumerate_dispatcher_helper<IntegralType, sizeof...(ints) - 1, ints...>::
        call(i, comparator, function, fallback, std::forward<Args>(args)...);
  }
};

// dispatch boolean to a lambda function
struct boolean_dispatcher {
  template <typename Lambda1, typename Lambda2, typename... Args>
  inline static void call(
      bool i,
      const Lambda1& function,
      const Lambda2& fallback,
      Args... args) {
    enumerate_dispatcher<bool, true, false>::call(
        i, function, fallback, std::forward<Args>(args)...);
  }
};

// A helper function that returns the last N-1 items of a tuple as a new tuple
template <typename TupleType, std::size_t... Is>
auto get_last_n_minus_one_impl(TupleType&& t, std::index_sequence<Is...>) {
  return std::tuple_cat(std::make_tuple(std::get<Is + 1>(t))...);
}

// A function that returns the last N-1 items of a tuple as a new tuple
template <typename TupleType>
auto get_last_n_minus_one(TupleType&& t) {
  // Get the size of the tuple
  constexpr auto size =
      std::tuple_size<typename std::remove_reference<TupleType>::type>::value;
  // Check if the size is greater than one
  return get_last_n_minus_one_impl(
      std::forward<TupleType>(t), std::make_index_sequence<size - 1>{});
}

template <
    typename TupleType,
    std::enable_if_t<std::tuple_size<TupleType>::value == 1, bool> = true>
auto get_last_n_minus_one(TupleType&& t) {
  return std::make_tuple();
}

template <
    typename IntegralTypesProcessed,
    typename IntegralTypesToProcess,
    typename Dispatchers>
struct product_dispatcher_helper;

template <typename... IntegralTypeProcessed>
struct product_dispatcher_helper<
    std::tuple<IntegralTypeProcessed...>,
    std::tuple<>,
    std::tuple<>> {
  template <typename Lambda1, typename Lambda2, typename... Args>
  inline static void call(
      std::tuple<>,
      std::tuple<IntegralTypeProcessed...> constants,
      std::tuple<>,
      const Lambda1& function,
      const Lambda2& fallback,
      Args... args) {
    function(constants, std::forward<Args>(args)...);
  }
};

template <
    typename... IntegralTypeProcessed,
    typename... IntegeralTypeToProcess,
    typename... Dispatcher>
struct product_dispatcher_helper<
    std::tuple<IntegralTypeProcessed...>,
    std::tuple<IntegeralTypeToProcess...>,
    std::tuple<Dispatcher...>> {
  template <typename Lambda1, typename Lambda2, typename... Args>
  inline static void call(
      std::tuple<Dispatcher...> dispatchers,
      std::tuple<IntegralTypeProcessed...> constants,
      std::tuple<IntegeralTypeToProcess...> integrals,
      const Lambda1& function,
      const Lambda2& fallback,
      Args... args) {
    std::get<0>(dispatchers)
        .call(
            std::get<0>(integrals),
            [&](auto i, Args... args) {
              auto new_dispatchers = get_last_n_minus_one(dispatchers);
              auto new_constants =
                  std::tuple_cat(constants, std::tuple<decltype(i)>(i));
              auto new_integrals = get_last_n_minus_one(integrals);
              product_dispatcher_helper<
                  decltype(new_constants),
                  decltype(new_integrals),
                  decltype(new_dispatchers)>::
                  call(
                      new_dispatchers,
                      new_constants,
                      new_integrals,
                      function,
                      fallback,
                      std::forward<Args>(args)...);
            },
            [&](auto i, Args... args) {
              fallback(
                  std::tuple_cat(constants, integrals),
                  std::forward<Args>(args)...);
            },
            std::forward<Args>(args)...);
  }
};

template <typename IntegralTypes, typename Dispatchers>
struct product_dispatcher;

// dispatch to a carsian product of a list of integers to a lambda function
template <typename... IntegeralType, typename... Dispatcher>
struct product_dispatcher<
    std::tuple<IntegeralType...>,
    std::tuple<Dispatcher...>> {
  template <typename Lambda1, typename Lambda2, typename... Args>
  inline static void call(
      std::tuple<IntegeralType...> integrals,
      const Lambda1& function,
      const Lambda2& fallback,
      Args... args) {
    static auto dispatchers = std::tuple<Dispatcher...>{};
    product_dispatcher_helper<
        std::tuple<>,
        std::tuple<IntegeralType...>,
        std::tuple<Dispatcher...>>::
        call(
            dispatchers,
            std::tuple<>{},
            integrals,
            function,
            fallback,
            std::forward<Args>(args)...);
  }
};

template <
    size_t N,
    typename IntegralType,
    typename Dispatcher,
    typename... OtherDispatchers>
struct union_dispatcher_helper {
  template <typename Lambda1, typename Lambda2, typename... Args>
  inline static void call(
      IntegralType i,
      const Lambda1& function,
      const Lambda2& fallback,
      Args... args) {
    Dispatcher::call(
        i,
        function,
        [&](auto i, Args... args) {
          union_dispatcher_helper<N - 1, IntegralType, OtherDispatchers...>::
              call(i, function, fallback, std::forward<Args>(args)...);
        },
        std::forward<Args>(args)...);
  }
};

template <typename IntegralType, typename Dispatcher>
struct union_dispatcher_helper<1, IntegralType, Dispatcher> {
  template <typename Lambda1, typename Lambda2, typename... Args>
  inline static void call(
      IntegralType i,
      const Lambda1& function,
      const Lambda2& fallback,
      Args... args) {
    Dispatcher::call(i, function, fallback, std::forward<Args>(args)...);
  }
};

// dispatch to a combination of a list of integers to a lambda function
template <typename IntegralType, typename... Dispatchers>
struct union_dispatcher {
  template <typename Lambda1, typename Lambda2, typename... Args>
  inline static void call(
      IntegralType i,
      const Lambda1& function,
      const Lambda2& fallback,
      Args... args) {
    union_dispatcher_helper<
        sizeof...(Dispatchers),
        IntegralType,
        Dispatchers...>::
        call(i, function, fallback, std::forward<Args>(args)...);
  }
};

#endif