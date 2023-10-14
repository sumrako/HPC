#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>

// Функция для умножения матрицы на CPU
void matrixMultiplyCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int matrix_size) {
    int size = matrix_size * matrix_size;
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < matrix_size; k++) {
                sum += A[i * matrix_size + k] * B[k * matrix_size + j];
            }
            C[i * matrix_size + j] = sum;
        }
    }
}

// Функция для умножения матрицы на GPU с использованием CUDA
__global__ void matrixMultiplyGPU(float* A, float* B, float* C, int matrix_size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < matrix_size && j < matrix_size) {
        float sum = 0.0f;
        for (int k = 0; k < matrix_size; k++) {
            sum += A[i * matrix_size + k] * B[k * matrix_size + j];
        }
        C[i * matrix_size + j] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Argument: " << argv[0] << " matrix size" << std::endl;
        return 1;
    }

    int matrix_size = atoi(argv[1]);
    if (matrix_size <= 0 || matrix_size > 2000) {
        std::cerr << "Not valid argument: " << argv[0] << " matrix size" << std::endl;
        return 1;
    }

    // Инициализация матриц A и B
    std::vector<float> A(matrix_size * matrix_size);
    std::vector<float> B(matrix_size * matrix_size);
    std::vector<float> C_CPU(matrix_size * matrix_size);
    std::vector<float> C_GPU(matrix_size * matrix_size);

    // Заполнение матриц случайными значениями
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < matrix_size * matrix_size; i++) {
        A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // Умножение на CPU и замер времени выполнения
    clock_t start_time = clock();
    matrixMultiplyCPU(A, B, C_CPU, matrix_size);
    clock_t end_time = clock();
    double cpu_execution_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    // Выделение памяти на GPU и копирование данных
    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeof(float) * matrix_size * matrix_size);
    cudaMalloc((void**)&d_B, sizeof(float) * matrix_size * matrix_size);
    cudaMalloc((void**)&d_C, sizeof(float) * matrix_size * matrix_size);

    cudaMemcpy(d_A, A.data(), sizeof(float) * matrix_size * matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeof(float) * matrix_size * matrix_size, cudaMemcpyHostToDevice);

    // Умножение на GPU и замер времени выполнения
    dim3 block_size(16, 16);
    dim3 grid_size((matrix_size + block_size.x - 1) / block_size.x, (matrix_size + block_size.y - 1) / block_size.y);

    start_time = clock();
    matrixMultiplyGPU<<<grid_size, block_size>>>(d_A, d_B, d_C, matrix_size);
    cudaDeviceSynchronize();
    end_time = clock();
    double gpu_execution_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    // Копирование результата с GPU на CPU
    cudaMemcpy(C_GPU.data(), d_C, sizeof(float) * matrix_size * matrix_size, cudaMemcpyDeviceToHost);

    // Освобождение памяти на GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Проверка результатов
    bool result_match = true;
    for (int i = 0; i < matrix_size * matrix_size; i++) {
        if (std::abs(C_CPU[i] - C_GPU[i]) > 1e-2) {
            result_match = false;
            break;
        }
    }

    if (result_match) {
        std::cout << "Results equals." << std::endl;
    } else {
        std::cout << "results not equals." << std::endl;
    }

    std::cout << "Time CPU: " << std::fixed << std::setprecision(10) << cpu_execution_time << " sec." << std::endl;
    std::cout << "Time GPU: " << std::fixed << std::setprecision(10) << gpu_execution_time << " sec." << std::endl;

    return 0;
}
