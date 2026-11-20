void zad1_1_2()
{
    float median(thrust::universal_vector<float> vec) {
    thrust::sort(thrust::device, vec.begin(), vec.end());
    return vec[vec.size() / 2];
  }
  
  int main() 
  {
      float k = 0.5;
      float ambient_temp = 20;
      thrust::universal_vector<float> temp{ 42, 24, 50 };
      auto transformation = [=] __host__ __device__ (float temp) { return temp + k * (ambient_temp - temp); };
  
      std::printf("step  median\n");
      for (int step = 0; step < 3; step++) {
          thrust::transform(thrust::device, temp.begin(), temp.end(), temp.begin(), transformation);
          float median_temp = median(temp);
          std::printf("%d     %.2f\n", step, median_temp);
      }
  }
}

void zad1_2_2()
{
    int main() {
      dli::where_am_I("CPU");
    
      thrust::universal_vector<int> vec{1};
      thrust::for_each(thrust::device, vec.begin(), vec.end(),
                       [] __host__ __device__(int) { dli::where_am_I("GPU"); });
    
      thrust::for_each(thrust::host, vec.begin(), vec.end(),
                       [] __host__ __device__(int) { dli::where_am_I("CPU"); });
    
      dli::where_am_I("CPU");
    }
}

void zad1_2_3()
{
    int main() {
        thrust::universal_vector<int> vec{ 1, 2, 3 };
        thrust::for_each(thrust::device, vec.begin(), vec.end(), []__device__(int val) {
            std::printf("printing %d on %s\n", val, dli::execution_space());
        });
    }

}


void zad1_2_4()
{
    float median(thrust::universal_vector<float> vec) {
      thrust::sort(thrust::device, vec.begin(), vec.end());
      return vec[vec.size() / 2];
    }
    
    int main() 
    {
        float k = 0.5;
        float ambient_temp = 20;
        thrust::universal_vector<float> temp{ 42, 24, 50 };
        auto transformation = [=] __host__ __device__ (float temp) { return temp + k * (ambient_temp - temp); };
    
        std::printf("step  median\n");
        for (int step = 0; step < 3; step++) {
            thrust::transform(thrust::device, temp.begin(), temp.end(), temp.begin(), transformation);
            float median_temp = median(temp);
            std::printf("%d     %.2f\n", step, median_temp);
        }
    }

}

void zad1_3_2()
{
    float variance(const thrust::universal_vector<float> &x, float mean) {
      // update the following line so that dereferencing `squared_differences`
      // returns `(xi - mean) * (xi - mean)`
      auto squared_differences = thrust::make_transform_iterator(
      x.begin(), [mean] __host__ __device__(float value) {
        return (value - mean) * (value - mean);
      });
    
      return thrust::reduce(thrust::device, squared_differences,
                            squared_differences + x.size()) /
             x.size();
    }
    
    float mean(thrust::universal_vector<float> vec) {
      return thrust::reduce(thrust::device, vec.begin(), vec.end()) / vec.size();
    }
    
    int main() {
      float ambient_temp = 20;
      thrust::universal_vector<float> prev{42, 24, 50};
      thrust::universal_vector<float> next{0, 0, 0};
    
      std::printf("step  variance\n");
      for (int step = 0; step < 3; step++) {
        thrust::transform(thrust::device, prev.begin(), prev.end(), next.begin(),
                          [=] __host__ __device__(float temp) {
                            return temp + 0.5 * (ambient_temp - temp);
                          });
        std::printf("%d     %.2f\n", step, variance(next, mean(next)));
        next.swap(prev);
      }
}
}

void zad1_4_2()
{
    void simulate(int height, int width,
                  const thrust::universal_vector<float> &in,
                        thrust::universal_vector<float> &out)
    {
      cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
    
      thrust::tabulate(thrust::device, out.begin(), out.end(), [temp_in] __host__ __device__(int id) {
        int column = id % temp_in.extent(1);
        int row    = id / temp_in.extent(1);
    
        if (row > 0 && column > 0 && row < temp_in.extent(0) - 1 && column < temp_in.extent(1) - 1)
        {
          float d2tdx2 = temp_in(row, column - 1) - 2 * temp_in(row, column) + temp_in(row, column + 1);
          float d2tdy2 = temp_in(row - 1, column) - 2 * temp_in(row, column) + temp_in(row + 1, column);
    
          return temp_in(row, column) + 0.2f * (d2tdx2 + d2tdy2);
        }
        else
        {
          return temp_in(row, column);
        }
      });
}


}

void zad5_1()
{
thrust::universal_vector<float>
row_temperatures(int height, int width, thrust::universal_vector<int> &row_ids,
                 thrust::universal_vector<float> &temp) {
  thrust::universal_vector<float> sums(height);

  // use counting and transform iterators to generates row numbers instead
  auto row_ids_begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [=] __host__ __device__(int i) { return i / width; });
  auto row_ids_end = row_ids_begin + temp.size();

  thrust::reduce_by_key(thrust::device, row_ids_begin, row_ids_end,
                        temp.begin(), thrust::make_discard_iterator(),
                        sums.begin());

  return sums;
}
}

void zad5_2()
{
struct mean_functor {
  int width;
  __host__ __device__ float operator()(float x) const { return x / width; }
};

thrust::universal_vector<float>
row_temperatures(int height, int width, thrust::universal_vector<int> &row_ids,
                 thrust::universal_vector<float> &temp) {
  thrust::universal_vector<float> means(height);
  auto means_output = thrust::make_transform_output_iterator(
      means.begin(), mean_functor{width});

  auto row_ids_begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [=] __host__ __device__(int i) { return i / width; });
  auto row_ids_end = row_ids_begin + temp.size();

  thrust::reduce_by_key(thrust::device, row_ids_begin, row_ids_end,
                        temp.begin(), thrust::make_discard_iterator(),
                        means_output);

  return means;
}
}

void zad6()
{
  int height = 4096;
  int width = 4096;

  thrust::device_vector<float> d_prev = dli::init(height, width);
  thrust::device_vector<float> d_next(height * width);
  thrust::host_vector<float> h_prev(height * width);

  for (int write_step = 0; write_step < 3; write_step++) {
    std::printf("   write step %d\n", write_step);
    thrust::copy(d_prev.begin(), d_prev.end(), h_prev.begin());
    dli::store(write_step, height, width, h_prev);

    for (int compute_step = 0; compute_step < 3; compute_step++) {
      auto begin = std::chrono::high_resolution_clock::now();
      dli::simulate(height, width, d_prev, d_next);
      auto end = std::chrono::high_resolution_clock::now();
      auto seconds = std::chrono::duration<double>(end - begin).count();
      std::printf("computed step %d in %g s\n", compute_step, seconds);
      d_prev.swap(d_next);
    }
}

void zad2_2_2():
{
        void simulate(int width, int height, const thrust::device_vector<float> &in,
                  thrust::device_vector<float> &out) {
      cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
      cub::DeviceTransform::Transform(
          thrust::make_counting_iterator(0), out.begin(), width * height,
          [=] __host__ __device__(int id) { return dli::compute(id, temp_in); });
    }
    
    
    int main()
    {
      int height = 2048;
      int width  = 8192;
    
      thrust::device_vector<float> d_prev = dli::init(height, width);
      thrust::device_vector<float> d_next(height * width);
      thrust::host_vector<float> h_prev(height * width);
    
      const int compute_steps = 500;
      const int write_steps = 3;
      for (int write_step = 0; write_step < write_steps; write_step++)
      {
        auto step_begin = std::chrono::high_resolution_clock::now();
        thrust::copy(d_prev.begin(), d_prev.end(), h_prev.begin());
    
        for (int compute_step = 0; compute_step < compute_steps; compute_step++)
        {
          simulate(width, height, d_prev, d_next);
          d_prev.swap(d_next);
        }
    
    
        auto write_begin = std::chrono::high_resolution_clock::now();
        dli::store(write_step, height, width, h_prev);
        auto write_end = std::chrono::high_resolution_clock::now();
        auto write_seconds = std::chrono::duration<double>(write_end - write_begin).count();
    
        auto step_end = std::chrono::high_resolution_clock::now();
        auto step_seconds = std::chrono::duration<double>(step_end - step_begin).count();
        std::printf("compute + write %d in %g s\n", write_step, step_seconds);
        std::printf("          write %d in %g s\n", write_step, write_seconds);
      }
    }
}

void zad2_3_2(){
    #include "dli.h"
    
    void simulate(int width, int height, const thrust::device_vector<float> &in,
                  thrust::device_vector<float> &out, cudaStream_t stream) {
      cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
      cub::DeviceTransform::Transform(
          thrust::make_counting_iterator(0), out.begin(), width * height,
          [=] __host__ __device__(int id) { return dli::compute(id, temp_in); },
          stream);
    }
    
    int main() {
      int height = 2048;
      int width = 8192;
    
      cudaStream_t compute_stream;
      cudaStreamCreate(&compute_stream);
    
      cudaStream_t copy_stream;
      cudaStreamCreate(&copy_stream);
    
      thrust::device_vector<float> d_prev = dli::init(height, width);
      thrust::device_vector<float> d_next(height * width);
      thrust::device_vector<float> d_buffer(height * width);
      thrust::host_vector<float> h_prev(height * width);
    
      const int compute_steps = 750;
      const int write_steps = 3;
      for (int write_step = 0; write_step < write_steps; write_step++) {
        cudaMemcpy(thrust::raw_pointer_cast(d_buffer.data()),
                   thrust::raw_pointer_cast(d_prev.data()),
                   height * width * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(thrust::raw_pointer_cast(h_prev.data()),
                        thrust::raw_pointer_cast(d_buffer.data()),
                        height * width * sizeof(float), cudaMemcpyDeviceToHost,
                        copy_stream);
    
        for (int compute_step = 0; compute_step < compute_steps; compute_step++) {
          simulate(width, height, d_prev, d_next, compute_stream);
          d_prev.swap(d_next);
        }
    
        cudaStreamSynchronize(copy_stream);
        dli::store(write_step, height, width, h_prev);
    
        cudaStreamSynchronize(compute_stream);
      }
    
      cudaStreamDestroy(compute_stream);
      cudaStreamDestroy(copy_stream);
    }
}


void zad2_4_2(){
    #include "dli.h"

    int main() {
      int height = 2048;
      int width = 8192;
    
      cudaStream_t compute_stream;
      cudaStreamCreate(&compute_stream);
    
      cudaStream_t copy_stream;
      cudaStreamCreate(&copy_stream);
    
      thrust::device_vector<float> d_prev = dli::init(height, width);
      thrust::device_vector<float> d_next(height * width);
      thrust::device_vector<float> d_buffer(height * width);
    
      thrust::universal_host_pinned_vector<float> h_prev(height * width);
    
      const int compute_steps = 750;
      const int write_steps = 3;
      for (int write_step = 0; write_step < write_steps; write_step++) {
        cudaMemcpy(thrust::raw_pointer_cast(d_buffer.data()),
                   thrust::raw_pointer_cast(d_prev.data()),
                   height * width * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(thrust::raw_pointer_cast(h_prev.data()),
                        thrust::raw_pointer_cast(d_buffer.data()),
                        height * width * sizeof(float), cudaMemcpyDeviceToHost,
                        copy_stream);
    
        for (int compute_step = 0; compute_step < compute_steps; compute_step++) {
          dli::simulate(width, height, d_prev, d_next, compute_stream);
          d_prev.swap(d_next);
        }
    
        cudaStreamSynchronize(copy_stream);
        dli::store(write_step, height, width, h_prev);
    
        cudaStreamSynchronize(compute_stream);
      }
    
      cudaStreamDestroy(compute_stream);
      cudaStreamDestroy(copy_stream);
    }
}

void zad3_2_2()
{
    __global__ void symmetry_check_kernel(dli::temperature_grid_f temp, int row) {
      int column = 0;
    
      if (abs(temp(row, column) - temp(temp.extent(0) - 1 - row, column)) > 0.1) {
        printf("Error: asymmetry in %d / %d\n", column, temp.extent(1));
      }
    }
    
    void symmetry_check(dli::temperature_grid_f temp, cudaStream_t stream) {
      int target_row = 0;
      symmetry_check_kernel<<<1, 1, 0, stream>>>(temp, target_row);
    }    
}

void zad3_2_3()
{
    __global__ void symmetry_check_kernel(dli::temperature_grid_f temp, int row) {
      int column = blockIdx.x * blockDim.x + threadIdx.x;
    
      if (abs(temp(row, column) - temp(temp.extent(0) - 1 - row, column)) > 0.1) {
        printf("Error: asymmetry in %d / %d\n", column, temp.extent(1));
      }
    }
    
    void symmetry_check(dli::temperature_grid_f temp, cudaStream_t stream) {
      int width = temp.extent(1);
      int block_size = 1024;
      int grid_size = cuda::ceil_div(width, block_size);
    
      int target_row = 0;
      symmetry_check_kernel<<<grid_size, block_size, 0, stream>>>(temp, target_row);
    }

}

void zad3_3_2()
{
    constexpr float bin_width = 10;
    
    __global__ void histogram_kernel(cuda::std::span<float> temperatures, 
                                     cuda::std::span<int> histogram)
    {
      int cell = blockIdx.x * blockDim.x + threadIdx.x;
      if (cell < temperatures.size()) {
        int bin = static_cast<int>(temperatures[cell] / bin_width);
    
          cuda::std::atomic_ref<int> ref(histogram[bin]);
          ref.fetch_add(1);
      }
    }
    
    
    void histogram(cuda::std::span<float> temperatures, 
                   cuda::std::span<int> histogram,
                   cudaStream_t stream)
    {
      int block_size = 256;
      int grid_size = cuda::ceil_div(temperatures.size(), block_size);
      histogram_kernel<<<grid_size, block_size, 0, stream>>>(
        temperatures, histogram);
    }
}

void zad3_4_2()
{
    constexpr float bin_width = 10;
    
    __global__ void histogram_kernel(cuda::std::span<float> temperatures,
                                     cuda::std::span<int> block_histograms,
                                     cuda::std::span<int> histogram) {
      cuda::std::span<int> block_histogram =
          block_histograms.subspan(blockIdx.x * histogram.size(), histogram.size());
    
      int cell = blockIdx.x * blockDim.x + threadIdx.x;
      int bin = static_cast<int>(temperatures[cell] / bin_width);
    
      cuda::atomic_ref<int, cuda::thread_scope_block> block_ref(
          block_histogram[bin]);
      block_ref.fetch_add(1);
      __syncthreads();
    
      if (threadIdx.x < histogram.size()) {
        cuda::atomic_ref<int, cuda::thread_scope_device> ref(
            histogram[threadIdx.x]);
        ref.fetch_add(block_histogram[threadIdx.x]);
      }
    }
    
    void histogram(cuda::std::span<float> temperatures,
                   cuda::std::span<int> block_histograms,
                   cuda::std::span<int> histogram, cudaStream_t stream) {
      int block_size = 256;
      int grid_size = cuda::ceil_div(temperatures.size(), block_size);
      histogram_kernel<<<grid_size, block_size, 0, stream>>>(
          temperatures, block_histograms, histogram);
    }
}

void zad3_5_2()
{
    
    __global__ void histogram_kernel(cuda::std::span<float> temperatures,
                                     cuda::std::span<int> histogram) {
      __shared__ int block_histogram[10];
    
      if (threadIdx.x < 10) {
        block_histogram[threadIdx.x] = 0;
      }
      __syncthreads();
    
      int cell = blockIdx.x * blockDim.x + threadIdx.x;
      int bin = static_cast<int>(temperatures[cell] / 10);
    
      cuda::atomic_ref<int, cuda::thread_scope_block> block_ref(
          block_histogram[bin]);
      block_ref.fetch_add(1);
      __syncthreads();
    
      if (threadIdx.x < 10) {
        cuda::atomic_ref<int, cuda::thread_scope_device> ref(
            histogram[threadIdx.x]);
        ref.fetch_add(block_histogram[threadIdx.x]);
      }
    }
    
    void histogram(cuda::std::span<float> temperatures,
                   cuda::std::span<int> block_histograms,
                   cuda::std::span<int> histogram, cudaStream_t stream) {
      int block_size = 256;
      int grid_size = cuda::ceil_div(temperatures.size(), block_size);
      histogram_kernel<<<grid_size, block_size, 0, stream>>>(temperatures,
                                                             histogram);
    }

}

void zad6_1_2()
{
    
    constexpr int block_size = 256;
    constexpr int items_per_thread = 1;
    constexpr int num_bins = 10;
    constexpr float bin_width = 10;
    
    __global__ void histogram_kernel(cuda::std::span<float> temperatures,
                                     cuda::std::span<int> histogram) {
      __shared__ int block_histogram[num_bins];
    
      int cell = blockIdx.x * blockDim.x + threadIdx.x;
      int bins[items_per_thread] = {
          static_cast<int>(temperatures[cell] / bin_width)};
    
      using histogram_t =
          cub::BlockHistogram<int, block_size, items_per_thread, num_bins,
                              cub::BlockHistogramAlgorithm::BLOCK_HISTO_ATOMIC>;
      __shared__ typename histogram_t::TempStorage temp_storage;
      histogram_t(temp_storage).Histogram(bins, block_histogram);
    
      __syncthreads();
      if (threadIdx.x < num_bins) {
        cuda::atomic_ref<int, cuda::thread_scope_device> ref(
            histogram[threadIdx.x]);
        ref.fetch_add(block_histogram[threadIdx.x]);
      }
    }
    
    void histogram(cuda::std::span<float> temperatures,
                   cuda::std::span<int> histogram, cudaStream_t stream) {
      int grid_size = cuda::ceil_div(temperatures.size(), block_size);
      histogram_kernel<<<grid_size, block_size, 0, stream>>>(temperatures,
                                                             histogram);
    }

}
