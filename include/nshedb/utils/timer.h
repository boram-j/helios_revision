#pragma once
#include <chrono>
#include <iostream>
#include <string>

namespace nshedb {
namespace utils {

struct Timer 
{
    std::chrono::high_resolution_clock::time_point time_start, time_end;
    std::chrono::microseconds time_diff;
    std::chrono::microseconds time_acc{0};
    std::string _name;

    Timer(std::string name) : _name(std::move(name)) {}

    void start() {
        time_start = std::chrono::high_resolution_clock::now();
    }
    void resume() {
        time_start = std::chrono::high_resolution_clock::now();
    }
    void pause() {
        time_end = std::chrono::high_resolution_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
        time_acc += time_diff;
    }
    void stop() {
        time_end = std::chrono::high_resolution_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
        time_acc += time_diff;
        std::cout << _name << " " << time_acc.count() << " us (" 
                  << static_cast<float>(time_acc.count()/ 1000000.0f) << " s)" << std::endl; 
        time_acc = std::chrono::microseconds(0);
    }
    void stop(int n) {
        time_acc = time_acc / n;
        std::cout << _name << " avg of " << n << " iterations: " 
                  << time_acc.count() << " us (" 
                  << static_cast<float>(time_acc.count()/ 1000000.0f) << " s)" << std::endl; 
    }
    void stop(int n, int ele) {
        time_acc = (time_acc / n) / ele;
        std::cout << _name << " avg of " << n << " iterations (amortize -- per ele) : " 
                  << time_acc.count() << " us (" 
                  << static_cast<float>(time_acc.count()/ 1000000.0f) << " s)" << std::endl; 
    }
    void print(int n) {
        time_acc = time_acc / n;
        std::cout << _name << " avg of " << n << " iterations: " 
                  << time_acc.count() << " us (" 
                  << static_cast<float>(time_acc.count()/ 1000000.0f) << " s)" << std::endl; 
    }
};

} // namespace utils
} // namespace nshedb