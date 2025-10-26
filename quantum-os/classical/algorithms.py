"""
Classical Algorithms Library

Standard classical algorithms that can run alongside quantum algorithms
"""

import numpy as np
from typing import List, Any, Callable, Dict, Optional


class ClassicalAlgorithms:
    """
    Library of classical algorithms

    Provides standard classical computing operations that complement
    quantum algorithms in the hybrid supercomputer
    """

    @staticmethod
    def linear_search(array: List[Any], target: Any) -> int:
        """
        Linear search - O(n)

        Args:
            array: Array to search
            target: Element to find

        Returns:
            Index or -1 if not found
        """
        for i, element in enumerate(array):
            if element == target:
                return i
        return -1

    @staticmethod
    def binary_search(array: List[Any], target: Any) -> int:
        """
        Binary search - O(log n)
        Requires sorted array

        Args:
            array: Sorted array
            target: Element to find

        Returns:
            Index or -1 if not found
        """
        left, right = 0, len(array) - 1

        while left <= right:
            mid = (left + right) // 2

            if array[mid] == target:
                return mid
            elif array[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return -1

    @staticmethod
    def quicksort(array: List[Any]) -> List[Any]:
        """
        Quicksort - O(n log n) average

        Args:
            array: Array to sort

        Returns:
            Sorted array
        """
        if len(array) <= 1:
            return array

        pivot = array[len(array) // 2]
        left = [x for x in array if x < pivot]
        middle = [x for x in array if x == pivot]
        right = [x for x in array if x > pivot]

        return ClassicalAlgorithms.quicksort(left) + middle + ClassicalAlgorithms.quicksort(right)

    @staticmethod
    def mergesort(array: List[Any]) -> List[Any]:
        """
        Mergesort - O(n log n) guaranteed

        Args:
            array: Array to sort

        Returns:
            Sorted array
        """
        if len(array) <= 1:
            return array

        mid = len(array) // 2
        left = ClassicalAlgorithms.mergesort(array[:mid])
        right = ClassicalAlgorithms.mergesort(array[mid:])

        return ClassicalAlgorithms._merge(left, right)

    @staticmethod
    def _merge(left: List[Any], right: List[Any]) -> List[Any]:
        """Merge two sorted arrays"""
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    @staticmethod
    def dijkstra(graph: Dict[Any, Dict[Any, float]], start: Any) -> Dict[Any, float]:
        """
        Dijkstra's shortest path algorithm - O(V² log V)

        Args:
            graph: Graph as adjacency dict
            start: Starting node

        Returns:
            Dict of shortest distances from start
        """
        import heapq

        distances = {node: float('inf') for node in graph}
        distances[start] = 0

        pq = [(0, start)]

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in graph.get(current_node, {}).items():
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

        return distances

    @staticmethod
    def dynamic_programming_knapsack(
        weights: List[float],
        values: List[float],
        capacity: float
    ) -> float:
        """
        0/1 Knapsack problem - O(nW)

        Args:
            weights: Item weights
            values: Item values
            capacity: Knapsack capacity

        Returns:
            Maximum value achievable
        """
        n = len(weights)
        capacity_int = int(capacity)

        dp = [[0 for _ in range(capacity_int + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(1, capacity_int + 1):
                weight_int = int(weights[i-1])

                if weight_int <= w:
                    dp[i][w] = max(
                        values[i-1] + dp[i-1][w-weight_int],
                        dp[i-1][w]
                    )
                else:
                    dp[i][w] = dp[i-1][w]

        return dp[n][capacity_int]

    @staticmethod
    def matrix_multiply_classical(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Classical matrix multiplication - O(n³)

        Args:
            A: First matrix
            B: Second matrix

        Returns:
            Product matrix
        """
        return np.dot(A, B)

    @staticmethod
    def fast_fourier_transform(signal: np.ndarray) -> np.ndarray:
        """
        Fast Fourier Transform - O(n log n)

        Args:
            signal: Time-domain signal

        Returns:
            Frequency-domain representation
        """
        return np.fft.fft(signal)

    @staticmethod
    def pagerank(
        graph: Dict[Any, List[Any]],
        damping: float = 0.85,
        max_iterations: int = 100
    ) -> Dict[Any, float]:
        """
        PageRank algorithm

        Args:
            graph: Graph as adjacency list
            damping: Damping factor
            max_iterations: Maximum iterations

        Returns:
            PageRank scores
        """
        nodes = list(graph.keys())
        n = len(nodes)

        # Initialize ranks
        ranks = {node: 1.0 / n for node in nodes}

        for _ in range(max_iterations):
            new_ranks = {}

            for node in nodes:
                rank_sum = sum(
                    ranks[incoming] / len(graph[incoming])
                    for incoming in nodes
                    if node in graph.get(incoming, [])
                )

                new_ranks[node] = (1 - damping) / n + damping * rank_sum

            ranks = new_ranks

        return ranks

    @staticmethod
    def monte_carlo_pi(num_samples: int) -> float:
        """
        Estimate π using Monte Carlo method

        Args:
            num_samples: Number of random samples

        Returns:
            Estimate of π
        """
        inside_circle = 0

        for _ in range(num_samples):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)

            if x*x + y*y <= 1:
                inside_circle += 1

        return 4.0 * inside_circle / num_samples

    @staticmethod
    def gradient_descent(
        function: Callable,
        gradient: Callable,
        initial: np.ndarray,
        learning_rate: float = 0.01,
        max_iterations: int = 1000
    ) -> np.ndarray:
        """
        Gradient descent optimization

        Args:
            function: Objective function
            gradient: Gradient function
            initial: Initial parameters
            learning_rate: Learning rate
            max_iterations: Maximum iterations

        Returns:
            Optimal parameters
        """
        params = initial.copy()

        for _ in range(max_iterations):
            grad = gradient(params)
            params -= learning_rate * grad

        return params
