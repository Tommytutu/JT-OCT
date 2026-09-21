// Native interval endpoint-elimination kernel for h=3 contracted JT-CG.
// Python only marshals contiguous arrays and dispatches terminal subproblems;
// aggregation, root joins, RMP maintenance and complete reduced-cost scans are
// performed here (C++/Gurobi).
#include "gurobi_c++.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#define API extern "C" __declspec(dllexport)
namespace {
thread_local std::string error;
struct EEMaster {
    int P,A; std::vector<int> root; GRBEnv env; GRBModel model;
    GRBConstr norm; std::vector<GRBConstr> rows; std::vector<GRBVar> var;
    std::vector<unsigned char> active; std::vector<int> ids; std::string output;
    EEMaster(int p,int a,const int* r,int method):P(p),A(a),root(r,r+p),env(true),model(start(env)),
      rows(a),var(2*p),active(2*p,0) {
        if(P<1||A<1) throw std::runtime_error("Invalid EE dimensions");
        for(int s=0;s<P;++s) if(root[s]<0||root[s]>=A) throw std::runtime_error("Invalid EE root map");
        model.set(GRB_IntParam_Threads,1); model.set(GRB_IntParam_Seed,20260905);
        model.set(GRB_IntParam_Method,method); model.set(GRB_IntParam_DualReductions,0);
        model.set(GRB_DoubleParam_FeasibilityTol,1e-9); model.set(GRB_DoubleParam_OptimalityTol,1e-9);
        norm=model.addConstr(GRBLinExpr()==1.);
        for(int r0=0;r0<A;++r0) rows[r0]=model.addConstr(GRBLinExpr()==0.);
    }
    static GRBEnv& start(GRBEnv& e){e.set(GRB_IntParam_OutputFlag,0);e.start();return e;}
    void update(int n,const int* input,const double* costs){
        for(int j=0;j<n;++j){int id=input[j]; if(id<0||id>=2*P||!std::isfinite(costs[j]))throw std::runtime_error("Invalid EE column");}
        for(int j=0;j<n;++j){int id=input[j],q=id/P,s=id%P;
            if(active[id]) var[id].set(GRB_DoubleAttr_Obj,costs[j]);
            else { GRBColumn col; if(q==0) col.addTerm(1.,norm); col.addTerm(q==0?1.:-1.,rows[root[s]]);
              var[id]=model.addVar(0.,GRB_INFINITY,costs[j],GRB_CONTINUOUS,col);active[id]=1;ids.push_back(id); }
        } model.update();
    }
    const char* solve(double seconds){
        model.set(GRB_DoubleParam_TimeLimit,std::max(1e-6,seconds));model.optimize();
        int code=model.get(GRB_IntAttr_Status);std::ostringstream os;os<<std::setprecision(17);
        os<<"{\"status\":"<<code<<",\"columns\":"<<ids.size()<<",\"rows\":"<<model.get(GRB_IntAttr_NumConstrs)
          <<",\"nonzeros\":"<<model.get(GRB_IntAttr_NumNZs);
        if(code==GRB_OPTIMAL){ double residual=std::abs(norm.get(GRB_DoubleAttr_Slack)),minrc=1e100;
          os<<",\"objective\":"<<model.get(GRB_DoubleAttr_ObjVal)<<",\"alpha\":"<<norm.get(GRB_DoubleAttr_Pi)<<",\"pi\":[";
          for(int r0=0;r0<A;++r0){if(r0)os<<",";double v=rows[r0].get(GRB_DoubleAttr_Pi);os<<v;residual=std::max(residual,std::abs(rows[r0].get(GRB_DoubleAttr_Slack)));}
          os<<"],\"support\":[";bool comma=false;for(int id:ids){minrc=std::min(minrc,var[id].get(GRB_DoubleAttr_RC));double x=var[id].get(GRB_DoubleAttr_X);if(x<=1e-9)continue;if(comma)os<<",";comma=true;os<<"["<<id<<","<<x<<"]";}
          os<<"],\"max_equality_residual\":"<<residual<<",\"minimum_active_reduced_cost\":"<<minrc;
          if(residual>1e-7||minrc<-1e-7)throw std::runtime_error("Native EE RMP numerical audit failed"); }
        os<<"}";output=os.str();return output.c_str();
    }
    int price(const double* cost,double alpha,const double* pi,double* rc,double* minimum) const {
        int count=0;minimum[0]=minimum[1]=std::numeric_limits<double>::infinity();
        for(int q=0;q<2;++q)for(int s=0;s<P;++s){int id=q*P+s;double c=cost[id];if(!std::isfinite(c)){rc[id]=std::numeric_limits<double>::infinity();continue;}double v=q==0?c-alpha-pi[root[s]]:c+pi[root[s]];rc[id]=v;minimum[q]=std::min(minimum[q],v);++count;}return count;
    }
};
inline bool finite(double x){return std::isfinite(x);}
}
API const char* ee_error(){return error.c_str();}
API void ee_sum_pairs(int P,const double* in,double* out){for(int s=0;s<P;++s){out[s]=in[s]+in[P+s];out[P+s]=in[2*P+s]+in[3*P+s];}}
API int ee_best_single(int P,const double* value,int* selected,double* best){*best=std::numeric_limits<double>::infinity();*selected=-1;for(int s=0;s<P;++s)if(value[s]<*best){*best=value[s];*selected=s;}return *selected>=0?0:1;}
API int ee_best_join(int P,int A,const int* root,const double* value,int* selected,double* best){
    std::vector<double> v(2*A,std::numeric_limits<double>::infinity());std::vector<int> id(2*A,-1);
    for(int q=0;q<2;++q)for(int s=0;s<P;++s){double x=value[q*P+s];int r=root[s];if(finite(x)&&x<v[q*A+r]){v[q*A+r]=x;id[q*A+r]=s;}}
    *best=std::numeric_limits<double>::infinity();selected[0]=selected[1]=-1;for(int r=0;r<A;++r)if(finite(v[r])&&finite(v[A+r])&&v[r]+v[A+r]<*best){*best=v[r]+v[A+r];selected[0]=id[r];selected[1]=id[A+r];}return selected[0]>=0?0:1;
}
// Complete native pricing admission.  Output ids are q*P+s, ordered by
// reduced cost then id independently in each retained block.
API int ee_admit(int P,const double* rc,const unsigned char* active,int cap,int* output){
    int n=0;for(int q=0;q<2;++q){std::vector<std::pair<double,int>> candidate;
      for(int s=0;s<P;++s){int id=q*P+s;if(!active[id]&&rc[id]<-1e-10)candidate.push_back({rc[id],id});}
      std::sort(candidate.begin(),candidate.end());int take=cap>0?std::min<int>(cap,candidate.size()):int(candidate.size());
      for(int j=0;j<take;++j)output[n++]=candidate[j].second;
    }return n;
}
API void* ee_create(int P,int A,const int* root,int method){try{error.clear();return new EEMaster(P,A,root,method);}catch(const GRBException& e){error=e.getMessage();return nullptr;}catch(const std::exception& e){error=e.what();return nullptr;}}
API int ee_update(void* p,int n,const int* ids,const double* costs){try{static_cast<EEMaster*>(p)->update(n,ids,costs);return 0;}catch(const GRBException& e){error=e.getMessage();return 1;}catch(const std::exception& e){error=e.what();return 1;}}
API const char* ee_solve(void* p,double seconds){try{return static_cast<EEMaster*>(p)->solve(seconds);}catch(const GRBException& e){error=e.getMessage();return nullptr;}catch(const std::exception& e){error=e.what();return nullptr;}}
API int ee_price(void* p,const double* cost,double alpha,const double* pi,double* rc,double* minimum){try{return static_cast<EEMaster*>(p)->price(cost,alpha,pi,rc,minimum);}catch(const std::exception& e){error=e.what();return -1;}}
API void ee_destroy(void* p){delete static_cast<EEMaster*>(p);}
