#include "SoA.h"
#include <chrono>

int main() {
    try {
        // Set up different experiment configurations
        std::vector<size_t> numPointsList = {10000, 100000}; // Example point sizes
        std::vector<int> clusters = {50,100}; // Different cluster counts
        std::vector<int> maxIterationsList = {50,100}; // Different max iterations
        std::vector<int> numThreadsList = {1,4,8}; // Example thread counts for parallel experiments

        // Loop over different configurations
        for (auto numPoints : numPointsList) {
            for (auto k : clusters) {
                for (auto maxIterations : maxIterationsList) {
                    std::string filename = "../data/dataset.csv"; // Path to your dataset

                    // Read data into SoA
                    PointsSoA points = readCSVToSoA(filename, numPoints);

                    // Start measuring time for sequential K-means
                    auto startSeq = std::chrono::high_resolution_clock::now();

                    // Perform sequential K-Means clustering
                    kmeansSoA_seq(points, numPoints, k, maxIterations);

                    // End measuring time for sequential
                    auto endSeq = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> durationSeq = endSeq - startSeq;

                    // Output the result for sequential K-means
                    std::cout << "Sequential execution time for " << numPoints << " points, "
                              << k << " clusters, and " << maxIterations << " max iterations: "
                              << durationSeq.count() << " seconds" << std::endl;
                    saveExecutionTimeToCSV("../data/execution_times_SoA.csv", numPoints, k, maxIterations, durationSeq.count(), 0);

                    // Loop over the number of threads for parallel version
                    for (auto numThreads : numThreadsList) {
                        // Set number of threads for parallel execution

                        // Start measuring time for parallel K-means
                        auto startPar = std::chrono::high_resolution_clock::now();

                        // Perform parallel K-Means clustering
                        kmeansSoA_parallel(points, numPoints, k, maxIterations, numThreads);

                        // End measuring time for parallel
                        auto endPar = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> durationPar = endPar - startPar;

                        // Output the result for parallel K-means
                        std::cout << "Parallel execution time with " << numThreads << " threads for "
                                  << numPoints << " points, " << k << " clusters, and "
                                  << maxIterations << " max iterations: " << durationPar.count() << " seconds" << std::endl;

                        // Save the execution time for parallel version
                        saveExecutionTimeToCSV("../data/execution_times_SoA.csv", numPoints, k, maxIterations, durationPar.count(), numThreads);
                    }
                }
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
