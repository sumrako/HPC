#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>

// Функция для сложения элементов вектора на CPU
double sumOnCPU(const std::vector<double>& data) {
    double sum = 0.0;
    for (double element : data) {
        sum += element;
    }
    return sum;
}

//Функция атомарного сложения double
__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Функция для сложения элементов вектора на GPU
__global__ void sumOnGPU(const double* data, int size, double* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    double localSum = 0.0;
    for (int i = tid; i < size; i += stride) {
        localSum += data[i];
    }

    // Вместо atomicAdd, используйте atomicAdd_double
    atomicAdd_double(result, localSum);
}


int main() {
    int vectorSize = 5000000000;
    std::vector<double> hostData(vectorSize);
    double *deviceData, *deviceResult;

    // Заполнение вектора случайными значениями
    for (int i = 0; i < vectorSize; ++i) {
        hostData[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // Засекаем время на CPU
    clock_t cpuStart = clock();
    double cpuSum = sumOnCPU(hostData);
    clock_t cpuEnd = clock();
    double cpuTime = (double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;

    // Выделение памяти на GPU
    cudaMalloc((void**)&deviceData, vectorSize * sizeof(double));
    cudaMalloc((void**)&deviceResult, sizeof(double));

    // Копируем данные с хоста на устройство
    cudaMemcpy(deviceData, hostData.data(), vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(deviceResult, 0, sizeof(double));
    
    dim3 blockDim(128);  // Количество потоков в блоке
    dim3 gridDim(256);  // Количество блоков в сетке

    
    // Засекаем время на GPU
    clock_t gpuStart = clock();
    sumOnGPU<<<gridDim, blockDim>>>(deviceData, vectorSize, deviceResult);
    cudaDeviceSynchronize();
    clock_t gpuEnd = clock();
    double gpuTime = (double)(gpuEnd - gpuStart) / CLOCKS_PER_SEC;

    // Копируем результат с устройства на хост
    double gpuSum;
    cudaMemcpy(&gpuSum, deviceResult, sizeof(double), cudaMemcpyDeviceToHost);

    // Освобождаем память на GPU
    cudaFree(deviceData);
    cudaFree(deviceResult);

    // Вывод результатов
    std::cout << "Sum on CPU: " << cpuSum << " (Time: " << cpuTime << " seconds)\n";
    std::cout << "Sum on GPU: " << gpuSum << " (Time: " << gpuTime << " seconds)\n";

    return 0;
}
