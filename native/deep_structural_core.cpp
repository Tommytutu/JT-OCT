// Generic D3-contracted junction-tree path core for D5--D7 experiments.
// State projection, endpoint elimination, min-sum recovery, RMP maintenance,
// complete pricing scans and memory preflight are native C++ operations.
#include "gurobi_c++.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <omp.h>

#define API extern "C" __declspec(dllexport)

namespace {
constexpr double INF=std::numeric_limits<double>::infinity();
thread_local std::string error;

struct Shape {
    int F,K,h; std::vector<int64_t> P;
    Shape(int f,int k,int hh):F(f),K(k),h(hh){
        if(F<1||K<1||h<1||h>4) throw std::runtime_error("Invalid deep structural dimensions");
        P.assign(h+1,1);
        for(int d=1;d<=h;++d){
            if(P[d-1]>(std::numeric_limits<int32_t>::max()-K)/F)
                throw std::runtime_error("Signature domain exceeds int32 indexing");
            P[d]=int64_t(F)*P[d-1]+K;
        }
    }
    int states()const{return int(P[h]);}
    int sep(int original_edge)const{
        int x=original_edge+1,v=0; while((x&1)==0){++v;x>>=1;} return h-v;
    }
    int project(int state,int depth)const{
        if(depth<1||depth>h||state<0||state>=P[h]) throw std::runtime_error("Invalid signature projection");
        int d=h,s=state,out=0,remaining=depth;
        for(int level=0;level<depth;++level){
            int64_t cut=int64_t(F)*P[d-1];
            if(s>=cut){
                int label=int(s-cut);
                out+=int(int64_t(F)*P[remaining-1]+label);
                return out;
            }
            int feature=int(s/P[d-1]); s=int(s%P[d-1]);
            out+=int(int64_t(feature)*P[remaining-1]);
            --d;--remaining;
        }
        return out;
    }
};

struct PathResult { double value=INF; std::vector<int> selected; };

PathResult path_opt(const Shape& shape,int blocks,int first,const double* cost){
    const int P=shape.states();
    if(blocks<1||first<0||first+blocks>(1<<shape.h)) throw std::runtime_error("Invalid retained path");
    std::vector<double> dp(P),next(P),group;
    std::vector<std::vector<int>> back(blocks,std::vector<int>(P,-1));
    std::copy(cost,cost+P,dp.begin());
    for(int q=1;q<blocks;++q){
        int depth=shape.sep(first+q-1),G=int(shape.P[depth]);
        group.assign(G,INF);std::vector<int> arg(G,-1);
        for(int s=0;s<P;++s) if(std::isfinite(dp[s])){
            int g=shape.project(s,depth);
            if(dp[s]<group[g]){group[g]=dp[s];arg[g]=s;}
        }
        #pragma omp parallel for schedule(static) if(P>32768)
        for(int s=0;s<P;++s){
            int g=shape.project(s,depth);double c=cost[int64_t(q)*P+s];
            next[s]=(std::isfinite(c)&&std::isfinite(group[g]))?c+group[g]:INF;
            back[q][s]=arg[g];
        }
        dp.swap(next);
    }
    PathResult out;auto it=std::min_element(dp.begin(),dp.end());
    if(it==dp.end()||!std::isfinite(*it))return out;
    out.value=*it;out.selected.assign(blocks,-1);out.selected.back()=int(it-dp.begin());
    for(int q=blocks-1;q>0;--q)out.selected[q-1]=back[q][out.selected[q]];
    return out;
}

struct Master {
    Shape shape;int B,first,P,R;uint64_t memory_limit;
    GRBEnv env;GRBModel model;std::vector<GRBConstr> norm,rows;
    std::vector<int> offsets;std::unordered_map<int,GRBVar> vars;std::string output;
    Master(int blocks,int f,int k,int h,int ff,int method,uint64_t limit):
      shape(f,k,h),B(blocks),first(ff),P(shape.states()),R(0),memory_limit(limit),
      env(true),model(start(env)){
        if(B<1||first<0||first+B>(1<<h))throw std::runtime_error("Invalid master path");
        if(int64_t(B)*P>INT32_MAX)throw std::runtime_error("INDEX_LIMIT: master column ids exceed int32");
        offsets.push_back(0);
        for(int e=0;e<B-1;++e){
            int64_t next=int64_t(R)+shape.P[shape.sep(first+e)];
            if(next>INT32_MAX)throw std::runtime_error("INDEX_LIMIT: separator rows exceed int32");
            R=int(next);offsets.push_back(R);
        }
        uint64_t fixed=uint64_t(B)*P*(sizeof(double)*2+sizeof(int32_t)+sizeof(uint8_t));
        if(limit&&fixed>limit)throw std::runtime_error("MEMORY_PRECHECK: dense cost domain exceeds limit");
        model.set(GRB_IntParam_Threads,1);model.set(GRB_IntParam_Seed,20260917);
        model.set(GRB_IntParam_Method,method);model.set(GRB_IntParam_DualReductions,0);
        model.set(GRB_DoubleParam_FeasibilityTol,1e-9);model.set(GRB_DoubleParam_OptimalityTol,1e-9);
        for(int q=0;q<B;++q)norm.push_back(model.addConstr(GRBLinExpr()==1.));
        for(int r=0;r<R;++r)rows.push_back(model.addConstr(GRBLinExpr()==0.));
    }
    static GRBEnv& start(GRBEnv& e){e.set(GRB_IntParam_OutputFlag,0);e.start();return e;}
    int row(int edge,int state)const{return offsets[edge]+shape.project(state,shape.sep(first+edge));}
    void update(int n,const int* ids,const double* costs){
        for(int j=0;j<n;++j){
            int id=ids[j],q=id/P,s=id%P;
            if(id<0||q>=B||!std::isfinite(costs[j]))throw std::runtime_error("Invalid deep master column");
            auto found=vars.find(id);
            if(found!=vars.end()){found->second.set(GRB_DoubleAttr_Obj,costs[j]);continue;}
            GRBColumn col;col.addTerm(1.,norm[q]);
            if(q>0)col.addTerm(-1.,rows[row(q-1,s)]);
            if(q<B-1)col.addTerm(1.,rows[row(q,s)]);
            vars.emplace(id,model.addVar(0.,GRB_INFINITY,costs[j],GRB_CONTINUOUS,col));
        }
        model.update();
    }
    const char* solve(double seconds){
        model.set(GRB_DoubleParam_TimeLimit,std::max(1e-6,seconds));model.optimize();
        int code=model.get(GRB_IntAttr_Status);std::ostringstream os;os<<std::setprecision(17);
        os<<"{\"status\":"<<code<<",\"columns\":"<<vars.size()<<",\"rows\":"<<model.get(GRB_IntAttr_NumConstrs)
          <<",\"nonzeros\":"<<model.get(GRB_IntAttr_NumNZs);
        if(code==GRB_OPTIMAL){
            std::vector<double> active(int64_t(B)*P,INF);double residual=0.,minrc=INF;
            os<<",\"objective\":"<<model.get(GRB_DoubleAttr_ObjVal)<<",\"alpha\":[";
            for(int q=0;q<B;++q){if(q)os<<",";os<<norm[q].get(GRB_DoubleAttr_Pi);residual=std::max(residual,std::abs(norm[q].get(GRB_DoubleAttr_Slack)));}
            os<<"],\"pi\":[";
            for(int r=0;r<R;++r){if(r)os<<",";os<<rows[r].get(GRB_DoubleAttr_Pi);residual=std::max(residual,std::abs(rows[r].get(GRB_DoubleAttr_Slack)));}
            for(auto& item:vars){active[item.first]=item.second.get(GRB_DoubleAttr_Obj);minrc=std::min(minrc,item.second.get(GRB_DoubleAttr_RC));}
            PathResult path=path_opt(shape,B,first,active.data());
            if(!std::isfinite(path.value)||std::abs(path.value-model.get(GRB_DoubleAttr_ObjVal))>1e-7)
                throw std::runtime_error("Native path recovery does not match RMP objective");
            os<<"],\"selection\":[";for(int q=0;q<B;++q){if(q)os<<",";os<<q*P+path.selected[q];}
            os<<"],\"max_equality_residual\":"<<residual<<",\"minimum_active_reduced_cost\":"<<minrc;
            if(residual>1e-7||minrc<-1e-7)throw std::runtime_error("Native deep RMP numerical audit failed");
        }
        os<<"}";output=os.str();return output.c_str();
    }
    int price(const double* costs,const double* alpha,const double* pi,double* rc,double* minima)const{
        int finite=0;
        #pragma omp parallel for schedule(static) reduction(+:finite) if(int64_t(B)*P>32768)
        for(int q=0;q<B;++q){minima[q]=INF;
            for(int s=0;s<P;++s){int id=q*P+s;double v=costs[id];
                if(!std::isfinite(v)){rc[id]=INF;continue;}v-=alpha[q];
                if(q>0)v+=pi[row(q-1,s)];
                if(q<B-1)v-=pi[row(q,s)];
                rc[id]=v;minima[q]=std::min(minima[q],v);++finite;
            }}return finite;
    }
};
}

API const char* ds_error(){return error.c_str();}
API int64_t ds_state_count(int F,int K,int h){try{return Shape(F,K,h).states();}catch(const std::exception& e){error=e.what();return -1;}}
API int ds_project(int F,int K,int h,int state,int depth){try{return Shape(F,K,h).project(state,depth);}catch(const std::exception& e){error=e.what();return -1;}}
API int ds_path_opt(int blocks,int F,int K,int h,int first,const double* costs,int* selected,double* value){
    try{auto out=path_opt(Shape(F,K,h),blocks,first,costs);*value=out.value;if(!std::isfinite(out.value))return 1;
        for(int q=0;q<blocks;++q)selected[q]=out.selected[q];return 0;}catch(const std::exception& e){error=e.what();return -1;}}
API int ds_eliminate_endpoints(int M,int P,const double* input,double* output){
    try{if(M<3||P<1)throw std::runtime_error("Endpoint elimination needs at least three blocks");
        int B=M-2;
        #pragma omp parallel for schedule(static) if(int64_t(B)*P>32768)
        for(int q=0;q<B;++q)for(int s=0;s<P;++s){double v=input[int64_t(q+1)*P+s];
            if(q==0)v+=input[s];if(q==B-1)v+=input[int64_t(M-1)*P+s];output[int64_t(q)*P+s]=v;}return 0;
    }catch(const std::exception& e){error=e.what();return 1;}}
API const char* ds_estimate(int M,int F,int K,int h,int eliminate){
    static thread_local std::string out;try{Shape s(F,K,h);int B=M-(eliminate?2:0);if(B<1)throw std::runtime_error("Invalid retained blocks");
        int64_t rows=B;for(int e=0;e<B-1;++e)rows+=s.P[s.sep((eliminate?1:0)+e)];
        uint64_t cols=uint64_t(B)*s.states();uint64_t bytes=cols*224ULL+uint64_t(rows)*160ULL;
        std::ostringstream os;os<<"{\"blocks\":"<<B<<",\"states_per_block\":"<<s.states()<<",\"columns\":"<<cols
          <<",\"rows\":"<<rows<<",\"estimated_peak_bytes\":"<<bytes<<"}";out=os.str();return out.c_str();
    }catch(const std::exception& e){error=e.what();return nullptr;}}
API void* ds_create(int B,int F,int K,int h,int first,int method,uint64_t memory_limit){
    try{error.clear();return new Master(B,F,K,h,first,method,memory_limit);}catch(const GRBException& e){error=e.getMessage();return nullptr;}catch(const std::exception& e){error=e.what();return nullptr;}}
API int ds_update(void* p,int n,const int* ids,const double* costs){try{static_cast<Master*>(p)->update(n,ids,costs);return 0;}catch(const GRBException& e){error=e.getMessage();return 1;}catch(const std::exception& e){error=e.what();return 1;}}
API const char* ds_solve(void* p,double seconds){try{return static_cast<Master*>(p)->solve(seconds);}catch(const GRBException& e){error=e.getMessage();return nullptr;}catch(const std::exception& e){error=e.what();return nullptr;}}
API int ds_price(void* p,const double* costs,const double* alpha,const double* pi,double* rc,double* minima){try{return static_cast<Master*>(p)->price(costs,alpha,pi,rc,minima);}catch(const std::exception& e){error=e.what();return -1;}}
API void ds_destroy(void* p){delete static_cast<Master*>(p);}

#include "deep_experiment.hpp"
