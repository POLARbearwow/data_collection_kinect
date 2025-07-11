#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class SafeQueue {
    std::queue<T> q_;
    std::mutex mtx_;
    std::condition_variable cv_;
public:
    void push(T &&value) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            q_.emplace(std::move(value));
        }
        cv_.notify_one();
    }
    bool pop(T &out) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [&]{ return !q_.empty(); });
        out = std::move(q_.front());
        q_.pop();
        return true;
    }
    bool try_pop(T &out) {
        std::lock_guard<std::mutex> lk(mtx_);
        if(q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        return true;
    }
    size_t size() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return q_.size();
    }
}; 