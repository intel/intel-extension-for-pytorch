#pragma once

#include <dnnl.h>
#include <dnnl.hpp>
#include <iostream>
#include <thread>
#include <mutex>
namespace dil {

class readwrite_lock 
{ 
public: 
    readwrite_lock() 
        : read_cnt(0) 
    { 
    } 
   
    void lock_read() 
    { 
        read_mtx.lock(); 
        if (++read_cnt == 1) 
            write_mtx.lock(); 
   
        read_mtx.unlock(); 
    } 
   
    void unlock_read() 
    { 
        read_mtx.lock(); 
        if (--read_cnt == 0) 
            write_mtx.unlock(); 
   
        read_mtx.unlock(); 
    } 
   
    void lock_write() 
    { 
        write_mtx.lock(); 
    } 
   
    void unlock_write() 
    { 
        write_mtx.unlock(); 
    } 
   
private: 
    std::mutex read_mtx; 
    std::mutex write_mtx; 
    int64_t read_cnt;
}; 


struct inner_product_params {
  dnnl::inner_product_forward::primitive_desc pd;
  dnnl::inner_product_forward prim;
  attr_t attr;
  attr_t src_attr;
  attr_t weights_attr;
  attr_t bias_attr;
  scale_t dst_scales;
  dims src_dims;
  inner_product_params( dnnl::inner_product_forward::primitive_desc p,  dnnl::inner_product_forward pri,
    attr_t op_a, attr_t src_a, attr_t weight_a, attr_t b_a, scale_t d, dims s_dims): pd(p), prim(pri), attr(op_a),
    src_attr(src_a), weights_attr(weight_a), bias_attr(b_a), dst_scales(d), src_dims(s_dims) {}
};


using  params = std::shared_ptr<inner_product_params>;


class ThreadPrimitiveCache {

public:
    int64_t get_cache_size(){
      return para_.size();
    }

    params& get_params(int64_t id){
      return para_.find(id)->second;
    }

    void insert_params(int64_t id, params p){
      para_.insert(std::make_pair(id, p));
    }

    bool hit(int64_t id){
      return para_.find(id) != para_.end();
    }

public:
  ThreadPrimitiveCache() : para_{} {}
  ~ThreadPrimitiveCache() = default;
  ThreadPrimitiveCache(const ThreadPrimitiveCache&) = default;
  ThreadPrimitiveCache& operator=(const ThreadPrimitiveCache&) = default;

private:
  std::map<int64_t, params> para_;
};

class PrimitiveCache{
public:
  static PrimitiveCache& singleton() {
    static  PrimitiveCache prim_para;
    return prim_para;
  }

public:
    int64_t get_cache_size(){
      return get_thread_param().get_cache_size();
    }

    params& get_params(int64_t id){
      return get_thread_param().get_params(id);
    }

    void insert_params(int64_t id, params p){
      get_thread_param().insert_params(id, p);
    }

    bool hit(int64_t id){
      return get_thread_param().hit(id);
    }

private:
  PrimitiveCache() {}
  ~PrimitiveCache() = default;
  PrimitiveCache(const PrimitiveCache&) = default;
  PrimitiveCache& operator=(const PrimitiveCache&) = default;

  ThreadPrimitiveCache& get_thread_param(){
    std::thread::id thread_id = std::this_thread::get_id();
    rw_mutex_.lock_read();
    auto it = params_map.find(thread_id);
    if (it != params_map.end()) {
      ThreadPrimitiveCache& ret = it->second;
      rw_mutex_.unlock_read();
      return ret;
    }
    rw_mutex_.unlock_read();
    
    rw_mutex_.lock_write();
    insert_params(thread_id);
    rw_mutex_.unlock_write();

    rw_mutex_.lock_read();
    ThreadPrimitiveCache& ret = params_map.find(thread_id)->second;
    rw_mutex_.unlock_read();
    return ret;
  }

  void insert_params(std::thread::id thread_id){
    auto thread_config = ThreadPrimitiveCache();
    params_map.insert(std::make_pair(thread_id, thread_config));
  }

private:
  std::map<std::thread::id, ThreadPrimitiveCache> params_map;
  readwrite_lock rw_mutex_;
};

}  // namespace dil