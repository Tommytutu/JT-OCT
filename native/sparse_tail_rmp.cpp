// Sparse complete-separator JT master. IDs are assigned only to generated rows/columns.
#include "gurobi_c++.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#define API extern "C" __declspec(dllexport)
namespace {
thread_local std::string error;
struct Master {
    GRBEnv env; GRBModel model;
    std::vector<GRBConstr> norm, sep;
    std::vector<GRBVar> vars;
    std::string output;
    static GRBEnv& start(GRBEnv& e){e.set(GRB_IntParam_OutputFlag,0);e.start();return e;}
    Master(int blocks,int method):env(true),model(start(env)){
        model.set(GRB_IntParam_Threads,1);model.set(GRB_IntParam_Seed,20260905);
        model.set(GRB_IntParam_Method,method);model.set(GRB_IntParam_DualReductions,0);
        model.set(GRB_DoubleParam_FeasibilityTol,1e-9);model.set(GRB_DoubleParam_OptimalityTol,1e-9);
        for(int i=0;i<blocks;++i)norm.push_back(model.addConstr(GRBLinExpr()==1.));
    }
    void update(int n,const int* blocks,const int* left,const int* right,const double* cost){
        int maxrow=-1;
        for(int j=0;j<n;++j){
            if(blocks[j]<0||blocks[j]>=norm.size()||left[j]<-1||right[j]<-1||!std::isfinite(cost[j]))
                throw std::runtime_error("Invalid sparse JT column");
            maxrow=std::max(maxrow,std::max(left[j],right[j]));
        }
        while(sep.size()<=maxrow)sep.push_back(model.addConstr(GRBLinExpr()==0.));
        model.update();
        for(int j=0;j<n;++j){
            GRBColumn c;c.addTerm(1.,norm[blocks[j]]);
            if(left[j]>=0)c.addTerm(-1.,sep[left[j]]);
            if(right[j]>=0)c.addTerm(1.,sep[right[j]]);
            vars.push_back(model.addVar(0.,GRB_INFINITY,cost[j],GRB_CONTINUOUS,c));
        }
        model.update();
    }
    const char* solve(double seconds){
        model.set(GRB_DoubleParam_TimeLimit,std::max(1e-6,seconds));model.optimize();
        int status=model.get(GRB_IntAttr_Status);std::ostringstream os;os<<std::setprecision(17);
        os<<"{\"status\":"<<status<<",\"columns\":"<<vars.size()<<",\"rows\":"<<norm.size()+sep.size();
        if(status==GRB_OPTIMAL){
            double residual=0.,minrc=1e100;
            os<<",\"objective\":"<<model.get(GRB_DoubleAttr_ObjVal)<<",\"alpha\":[";
            for(size_t i=0;i<norm.size();++i){if(i)os<<",";os<<norm[i].get(GRB_DoubleAttr_Pi);
                residual=std::max(residual,std::abs(norm[i].get(GRB_DoubleAttr_Slack)));}
            os<<"],\"pi\":[";
            for(size_t i=0;i<sep.size();++i){if(i)os<<",";os<<sep[i].get(GRB_DoubleAttr_Pi);
                residual=std::max(residual,std::abs(sep[i].get(GRB_DoubleAttr_Slack)));}
            os<<"],\"support\":[";bool comma=false;
            for(size_t i=0;i<vars.size();++i){
                minrc=std::min(minrc,vars[i].get(GRB_DoubleAttr_RC));double x=vars[i].get(GRB_DoubleAttr_X);
                if(x<=1e-9)continue;if(comma)os<<",";comma=true;os<<"["<<i<<","<<x<<"]";
            }
            if(residual>1e-7||minrc<-1e-7)throw std::runtime_error("Sparse master numerical audit failed");
            os<<"],\"max_equality_residual\":"<<residual<<",\"minimum_active_reduced_cost\":"<<minrc;
        }
        os<<"}";output=os.str();return output.c_str();
    }
};
}
API const char* strmp_error(){return error.c_str();}
API void* strmp_create(int blocks,int method){
    try{error.clear();if(blocks<1)throw std::runtime_error("Invalid number of blocks");return new Master(blocks,method);}
    catch(const GRBException& e){error=e.getMessage();return nullptr;}catch(const std::exception& e){error=e.what();return nullptr;}
}
API int strmp_update(void* p,int n,const int* blocks,const int* left,const int* right,const double* cost){
    try{static_cast<Master*>(p)->update(n,blocks,left,right,cost);return 0;}
    catch(const GRBException& e){error=e.getMessage();return 1;}catch(const std::exception& e){error=e.what();return 1;}
}
API const char* strmp_solve(void* p,double seconds){
    try{return static_cast<Master*>(p)->solve(seconds);}
    catch(const GRBException& e){error=e.getMessage();return nullptr;}catch(const std::exception& e){error=e.what();return nullptr;}
}
API void strmp_destroy(void* p){delete static_cast<Master*>(p);}
