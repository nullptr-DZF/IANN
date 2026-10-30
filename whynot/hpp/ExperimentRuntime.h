#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef _WIN32
#include <csignal>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace iann_runtime {
inline std::string json_quote(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        switch(c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if(c<0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
            else out << c;
        }
    }
    out << '"'; return out.str();
}
inline std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> fields; std::stringstream in(line); std::string item;
    while(std::getline(in,item,',')) fields.push_back(item);
    if(!line.empty() && line.back()==',') fields.push_back("");
    return fields;
}
inline std::string trim_cr(std::string line) {
    if(!line.empty() && line.back()=='\r') line.pop_back(); return line;
}
struct Point {
    size_t query=0, expected=0, rank=0, ef=0, k=0;
};
inline std::vector<Point> read_keys(const std::string& path) {
    std::ifstream in(path); if(!in) throw std::runtime_error("cannot open keys: "+path);
    std::string line; if(!std::getline(in,line)) throw std::runtime_error("missing keys header");
    auto names=split(trim_cr(line)); std::map<std::string,size_t> columns;
    for(size_t i=0;i<names.size();++i) {
        if(!columns.emplace(names[i],i).second) throw std::runtime_error("duplicate keys column");
    }
    for(const char* key : {"query_id","expected_id","ground_truth_rank"})
        if(!columns.count(key)) throw std::runtime_error(std::string("missing keys column: ")+key);
    std::vector<Point> points; std::set<std::pair<size_t,size_t>> seen;
    while(std::getline(in,line)) {
        line=trim_cr(line); if(line.empty()) continue;
        auto fields=split(line); if(fields.size()!=names.size()) throw std::runtime_error("malformed keys row");
        Point p;
        p.query=std::stoull(fields[columns.at("query_id")]);
        p.expected=std::stoull(fields[columns.at("expected_id")]);
        p.rank=std::stoull(fields[columns.at("ground_truth_rank")]);
        if(columns.count("ef_best") && !fields[columns.at("ef_best")].empty()) p.ef=std::stoull(fields[columns.at("ef_best")]);
        if(columns.count("k_best") && !fields[columns.at("k_best")].empty()) p.k=std::stoull(fields[columns.at("k_best")]);
        if(!p.rank || !seen.emplace(p.query,p.expected).second) throw std::runtime_error("invalid or duplicate point");
        points.push_back(p);
    }
    return points;
}
inline std::vector<float> read_queries(const std::string& path,size_t& dimension) {
    std::ifstream in(path,std::ios::binary); int d=0;
    if(!in || !in.read(reinterpret_cast<char*>(&d),4) || d<=0) throw std::runtime_error("bad query fvecs: "+path);
    dimension=size_t(d); in.seekg(0,std::ios::end); auto end=in.tellg();
    if(end<0 || size_t(end)%(4*(dimension+1))) throw std::runtime_error("truncated query fvecs");
    in.seekg(0); std::vector<float> values(size_t(end)/(4*(dimension+1))*dimension);
    for(size_t row=0;row<values.size()/dimension;++row) {
        in.read(reinterpret_cast<char*>(&d),4);
        if(d!=int(dimension)) throw std::runtime_error("mixed query dimensions");
        in.read(reinterpret_cast<char*>(values.data()+row*dimension),4*dimension);
    }
    if(!in) throw std::runtime_error("query read failed"); return values;
}
inline double milliseconds(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
}

// A hard timeout appends an already serialized terminal row, then exits so that
// the supervisor can restart remaining points without reusing interrupted state.
class BudgetGuard {
#ifndef _WIN32
    static int& output_fd() { static int fd=-1; return fd; }
    static int& progress_fd() { static int fd=-1; return fd; }
    static std::string& timeout_row() { static std::string row; return row; }
    static const char*& row_data() { static const char* p=nullptr; return p; }
    static size_t& row_size() { static size_t n=0; return n; }
    static void expired(int) {
        if(output_fd()>=0 && row_data()) {
            const char* data=row_data(); size_t remaining=row_size();
            while(remaining) { ssize_t n=::write(output_fd(),data,remaining); if(n<=0) break; data+=n; remaining-=size_t(n); }
        }
        if(progress_fd()>=0) { const char marker[]="TIMEOUT\n"; ::write(progress_fd(),marker,sizeof(marker)-1); }
        ::_exit(124);
    }
#endif
public:
    BudgetGuard(const std::string& output,const std::string& progress) {
#ifndef _WIN32
        output_fd()=::open(output.c_str(),O_WRONLY|O_APPEND);
        progress_fd()=::open(progress.c_str(),O_WRONLY|O_APPEND);
        if(output_fd()<0 || progress_fd()<0) throw std::runtime_error("cannot open timeout descriptors");
        std::signal(SIGALRM,expired);
#else
        (void)output; (void)progress;
#endif
    }
    void arm(double ms,const std::string& row) {
#ifndef _WIN32
        timeout_row()=row; row_data()=timeout_row().data(); row_size()=timeout_row().size();
        struct itimerval timer={};
        long long micros=std::max<long long>(1,static_cast<long long>(ms*1000));
        timer.it_value.tv_sec=micros/1000000; timer.it_value.tv_usec=micros%1000000;
        setitimer(ITIMER_REAL,&timer,nullptr);
#else
        (void)ms; (void)row;
#endif
    }
    void cancel() {
#ifndef _WIN32
        struct itimerval timer={}; setitimer(ITIMER_REAL,&timer,nullptr); row_data()=nullptr; row_size()=0;
#endif
    }
    ~BudgetGuard() {
        cancel();
#ifndef _WIN32
        if(output_fd()>=0) ::close(output_fd()); if(progress_fd()>=0) ::close(progress_fd());
#endif
    }
};
}
