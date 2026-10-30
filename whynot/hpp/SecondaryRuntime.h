#pragma once
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#ifndef _WIN32
#include <csignal>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace iann_secondary {
#ifndef _WIN32
static int progress_fd=-1,checkpoint_fd=-1;
static char timeout_record[160];
static volatile sig_atomic_t timeout_length=0;
inline void timeout_handler(int) {
    if(progress_fd>=0) ::write(progress_fd,timeout_record,size_t(timeout_length));
    if(checkpoint_fd>=0) ::write(checkpoint_fd,timeout_record,size_t(timeout_length));
    _exit(124);
}
#endif

inline std::string env(const char* key,const std::string& fallback="") {
    const char* value=std::getenv(key);return value && *value?value:fallback;
}
inline std::string json_string(const std::string& text) {
    std::string out="\"";
    const char* hex="0123456789abcdef";
    for(unsigned char c:text) {
        if(c=='"' || c=='\\') {out+='\\';out+=char(c);}
        else if(c<32) {out+="\\u00";out+=hex[c>>4];out+=hex[c&15];}
        else out+=char(c);
    }
    return out+'"';
}

class Runtime {
    std::ofstream progress,errors,checkpoint;
    std::chrono::steady_clock::time_point started;
    size_t query=0,expected=0;
    double budget;
    bool replaying=false;
#ifndef _WIN32
    int owned_checkpoint_fd=-1;
#endif
    void event(const char* status,bool persist_failure=false) {
        if(progress.is_open()) {progress<<status<<'\t'<<query<<'\t'<<expected<<'\n';progress.flush();}
        if(persist_failure && !replaying && checkpoint.is_open()) {
            checkpoint<<status<<'\t'<<query<<'\t'<<expected<<'\n';checkpoint.flush();
        }
    }
    void disarm() {
#ifndef _WIN32
        struct itimerval timer={};setitimer(ITIMER_REAL,&timer,nullptr);
#endif
    }
public:
    Runtime():budget(std::stod(env("EXPERIMENT_BUDGET_MS","600000"))) {
        if(!std::isfinite(budget) || budget<=0) throw std::invalid_argument("invalid point budget");
        const std::string p=env("EXPERIMENT_PROGRESS"),e=env("EXPERIMENT_ERRORS"),c=env("EXPERIMENT_REPAIR_STATE");
        if(!p.empty()) {progress.open(p,std::ios::app);if(!progress)throw std::runtime_error("cannot open progress file");}
        if(!e.empty()) {errors.open(e,std::ios::app);if(!errors)throw std::runtime_error("cannot open error file");}
        if(!c.empty()) {checkpoint.open(c,std::ios::app);if(!checkpoint)throw std::runtime_error("cannot open repair state");}
#ifndef _WIN32
        if(!p.empty()) progress_fd=::open(p.c_str(),O_WRONLY|O_CREAT|O_APPEND,0644);
        if(!c.empty()) owned_checkpoint_fd=checkpoint_fd=::open(c.c_str(),O_WRONLY|O_CREAT|O_APPEND,0644);
        if((!p.empty() && progress_fd<0)||(!c.empty() && checkpoint_fd<0))throw std::runtime_error("cannot open timeout event file");
        std::signal(SIGALRM,timeout_handler);
#endif
    }
    ~Runtime() {
        disarm();
#ifndef _WIN32
        if(progress_fd>=0) {::close(progress_fd);progress_fd=-1;}
        if(owned_checkpoint_fd>=0) {::close(owned_checkpoint_fd);checkpoint_fd=-1;}
#endif
    }
    void begin(size_t q,size_t e,bool replay=false) {
        disarm();query=q;expected=e;replaying=replay;event("BEGIN");started=std::chrono::steady_clock::now();
#ifndef _WIN32
        checkpoint_fd=replay?-1:owned_checkpoint_fd;
        // Format outside the handler: only write and _exit are used on timeout.
        timeout_length=std::snprintf(timeout_record,sizeof(timeout_record),"TIMEOUT\t%zu\t%zu\n",query,expected);
        struct itimerval timer={};
        timer.it_value.tv_sec=long(budget/1000);
        timer.it_value.tv_usec=long(std::fmod(budget,1000.0)*1000);
        if(!timer.it_value.tv_sec && !timer.it_value.tv_usec)timer.it_value.tv_usec=1;
        setitimer(ITIMER_REAL,&timer,nullptr);
#endif
    }
    bool finish() {
        disarm();const bool timeout=elapsed_ms()>=budget;
        event(timeout?"TIMEOUT":"END",timeout);return timeout;
    }
    double elapsed_ms() const {
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    }
    double budget_ms() const {return budget;}
    void fail(const std::string& message) {
        disarm();event("EXCEPTION",true);
        if(errors.is_open()) {
            errors<<"{\"query_id\":"<<query<<",\"expected_id\":"<<expected
                  <<",\"error\":"<<json_string(message)<<"}\n";errors.flush();
        }
        std::cerr<<"point exception: "<<message<<'\n';
    }
};
}
