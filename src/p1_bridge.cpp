#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "defines.hpp"
#include "quantization/config.h"
#include "quantization/saq_data.hpp"
#include "quantization/saq_estimator.hpp"
#include "quantization/saq_quantizer.hpp"
#include "quantization/single_data.hpp"
#include "utils/BS_thread_pool.hpp"
#include "utils/tools.hpp"

namespace {

using namespace saqlib;

thread_local std::string last_error;

class SaqP1Context {
  public:
    SaqP1Context(
        const float *variance,
        std::size_t dimension,
        float average_bits,
        int adjustment_rounds,
        unsigned int seed)
        : dimension_(dimension) {
        if (variance == nullptr || dimension == 0) {
            throw std::invalid_argument("variance must be non-null and dimension must be positive");
        }
        if (average_bits < 0) {
            throw std::invalid_argument("average_bits must be non-negative");
        }

        QuantizeConfig config;
        config.avg_bits = average_bits;
        config.enable_segmentation = true;
        config.single.random_rotation = true;
        config.single.use_fastscan = false;
        config.single.caq_adj_rd_lmt = adjustment_rounds;

        std::srand(seed);
        FloatVec variance_vector = Eigen::Map<const FloatVec>(variance, dimension);
        SaqDataMaker data_maker(config, dimension);
        data_maker.set_variance(std::move(variance_vector));
        data_ = data_maker.return_data();

        memory_size_ = SaqSingleDataWrapper::calculate_memory_size(data_->quant_plan);
        stride_ = utils::rd_up_to_multiple_of(memory_size_, std::size_t{64});
        build_plan_metadata();
    }

    ~SaqP1Context() {
        std::free(encoded_memory_);
    }

    SaqP1Context(const SaqP1Context &) = delete;
    SaqP1Context &operator=(const SaqP1Context &) = delete;

    double encode(
        const float *vectors,
        std::size_t vector_count,
        std::size_t dimension,
        std::size_t thread_count) {
        if (vectors == nullptr && vector_count != 0) {
            throw std::invalid_argument("vectors must be non-null");
        }
        if (dimension != dimension_) {
            throw std::invalid_argument("encode dimension does not match the trained SAQ context");
        }
        if (stride_ != 0 && vector_count > std::numeric_limits<std::size_t>::max() / stride_) {
            throw std::overflow_error("encoded SAQ allocation size overflow");
        }

        std::free(encoded_memory_);
        encoded_memory_ = nullptr;
        encoded_count_ = vector_count;

        const std::size_t total_bytes = stride_ * vector_count;
        if (total_bytes != 0) {
            encoded_memory_ = static_cast<std::uint8_t *>(std::aligned_alloc(64, total_bytes));
            if (encoded_memory_ == nullptr) {
                throw std::bad_alloc();
            }
            std::memset(encoded_memory_, 0, total_bytes);
        }

        const auto start = std::chrono::steady_clock::now();
        if (vector_count != 0) {
            BS::thread_pool pool(normalize_thread_count(thread_count, vector_count));
            auto futures = pool.submit_blocks(
                std::size_t{0},
                vector_count,
                [this, vectors](std::size_t begin, std::size_t end) {
                    SAQuantizerSingle quantizer(data_.get());
                    SaqSingleDataWrapper wrapper(data_->quant_plan);
                    for (std::size_t index = begin; index < end; ++index) {
                        wrapper.set_memory_base(encoded_vector(index));
                        FloatVec vector = Eigen::Map<const FloatVec>(
                            vectors + index * dimension_, dimension_);
                        quantizer.quantize(vector, &wrapper);
                    }
                });
            futures.get();
        }
        const auto stop = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(stop - start).count();
    }

    void distances(
        const float *queries,
        std::size_t query_count,
        std::size_t dimension,
        const std::int64_t *candidate_indices,
        std::size_t candidates_per_query,
        std::size_t thread_count,
        float *output) const {
        if (encoded_memory_ == nullptr && encoded_count_ != 0) {
            throw std::runtime_error("encoded SAQ data is unavailable");
        }
        if ((queries == nullptr || candidate_indices == nullptr || output == nullptr) &&
            query_count * candidates_per_query != 0) {
            throw std::invalid_argument("distance inputs must be non-null");
        }
        if (dimension != dimension_) {
            throw std::invalid_argument("query dimension does not match the trained SAQ context");
        }

        const std::size_t pair_count = query_count * candidates_per_query;
        for (std::size_t pair = 0; pair < pair_count; ++pair) {
            const std::int64_t index = candidate_indices[pair];
            if (index < 0 || static_cast<std::size_t>(index) >= encoded_count_) {
                throw std::out_of_range("candidate index is outside the encoded base set");
            }
        }

        if (query_count == 0) {
            return;
        }

        SearcherConfig searcher_config;
        searcher_config.dist_type = DistType::L2Sqr;
        BS::thread_pool pool(normalize_thread_count(thread_count, query_count));
        auto futures = pool.submit_blocks(
            std::size_t{0},
            query_count,
            [this, queries, candidate_indices, candidates_per_query, output, searcher_config](
                std::size_t begin,
                std::size_t end) {
                SaqSingleDataWrapper wrapper(data_->quant_plan);
                for (std::size_t query_index = begin; query_index < end; ++query_index) {
                    FloatVec query = Eigen::Map<const FloatVec>(
                        queries + query_index * dimension_, dimension_);
                    SaqSingleEstimator<DistType::L2Sqr> estimator(
                        *data_, searcher_config, query);
                    for (std::size_t candidate = 0; candidate < candidates_per_query; ++candidate) {
                        const std::size_t pair = query_index * candidates_per_query + candidate;
                        const std::size_t data_index = static_cast<std::size_t>(candidate_indices[pair]);
                        wrapper.set_memory_base(encoded_vector(data_index));
                        output[pair] = estimator.compAccurateDist(wrapper);
                    }
                }
            });
        futures.get();
    }

    std::size_t code_bits() const {
        return code_bits_;
    }

    std::size_t active_segments() const {
        return active_segments_;
    }

    std::size_t memory_size() const {
        return memory_size_;
    }

    const std::string &plan() const {
        return plan_;
    }

  private:
    static std::size_t normalize_thread_count(
        std::size_t requested,
        std::size_t work_items) {
        return std::max(
            std::size_t{1},
            std::min(requested == 0 ? std::size_t{1} : requested, work_items));
    }

    std::uint8_t *encoded_vector(std::size_t index) const {
        return encoded_memory_ + index * stride_;
    }

    void build_plan_metadata() {
        std::size_t remaining_dimensions = dimension_;
        std::size_t offset = 0;
        std::ostringstream stream;
        for (const auto &[segment_dimensions, bits] : data_->quant_plan) {
            const std::size_t used_dimensions = std::min(
                segment_dimensions, remaining_dimensions);
            if (bits > 0) {
                code_bits_ += used_dimensions * bits;
                ++active_segments_;
            }
            stream << "[" << offset << "," << offset + used_dimensions << "):" << bits;
            remaining_dimensions -= used_dimensions;
            offset += used_dimensions;
            if (remaining_dimensions != 0) {
                stream << ",";
            }
        }
        plan_ = stream.str();
    }

    const std::size_t dimension_;
    std::unique_ptr<SaqData> data_;
    std::uint8_t *encoded_memory_ = nullptr;
    std::size_t encoded_count_ = 0;
    std::size_t memory_size_ = 0;
    std::size_t stride_ = 0;
    std::size_t code_bits_ = 0;
    std::size_t active_segments_ = 0;
    std::string plan_;
};

template <typename Function>
int guarded_call(Function &&function) noexcept {
    try {
        last_error.clear();
        function();
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    } catch (...) {
        last_error = "unknown C++ exception";
        return -1;
    }
}

} // namespace

extern "C" {

const char *saq_p1_last_error() noexcept {
    return last_error.c_str();
}

void *saq_p1_create(
    const float *variance,
    std::size_t dimension,
    float average_bits,
    int adjustment_rounds,
    unsigned int seed) noexcept {
    try {
        last_error.clear();
        return new SaqP1Context(
            variance, dimension, average_bits, adjustment_rounds, seed);
    } catch (const std::exception &error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "unknown C++ exception";
        return nullptr;
    }
}

void saq_p1_destroy(void *context) noexcept {
    delete static_cast<SaqP1Context *>(context);
}

int saq_p1_encode(
    void *context,
    const float *vectors,
    std::size_t vector_count,
    std::size_t dimension,
    std::size_t thread_count,
    double *elapsed_seconds) noexcept {
    return guarded_call([&] {
        if (context == nullptr || elapsed_seconds == nullptr) {
            throw std::invalid_argument("context and elapsed_seconds must be non-null");
        }
        *elapsed_seconds = static_cast<SaqP1Context *>(context)->encode(
            vectors, vector_count, dimension, thread_count);
    });
}

int saq_p1_distances(
    void *context,
    const float *queries,
    std::size_t query_count,
    std::size_t dimension,
    const std::int64_t *candidate_indices,
    std::size_t candidates_per_query,
    std::size_t thread_count,
    float *output) noexcept {
    return guarded_call([&] {
        if (context == nullptr) {
            throw std::invalid_argument("context must be non-null");
        }
        static_cast<SaqP1Context *>(context)->distances(
            queries,
            query_count,
            dimension,
            candidate_indices,
            candidates_per_query,
            thread_count,
            output);
    });
}

std::size_t saq_p1_code_bits(void *context) noexcept {
    return context == nullptr ? 0 : static_cast<SaqP1Context *>(context)->code_bits();
}

std::size_t saq_p1_active_segments(void *context) noexcept {
    return context == nullptr ? 0 : static_cast<SaqP1Context *>(context)->active_segments();
}

std::size_t saq_p1_memory_size(void *context) noexcept {
    return context == nullptr ? 0 : static_cast<SaqP1Context *>(context)->memory_size();
}

const char *saq_p1_plan(void *context) noexcept {
    return context == nullptr ? "" : static_cast<SaqP1Context *>(context)->plan().c_str();
}

} // extern "C"
