// Persistent Gurobi RMP for a D3-contracted D4/D5 junction tree.
// Two/four blocks retain the complete ancestor separators. Private D3 costs
// arrive from exact CPU/GPU oracles; feasible upper costs may decrease in place.
#include "gurobi_c++.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#define API extern "C" __declspec(dllexport)
namespace {
thread_local std::string error;
struct Master {
    int M,F,K,A,P,R;
    GRBEnv env;GRBModel model;
    std::vector<GRBConstr> norm,sep;
    std::vector<GRBVar> vars;
    std::vector<unsigned char> row,active;
    std::vector<int> indices;
    std::string output;
    Master(int m,int f,int k,int method):M(m),F(f),K(k),A(f+k),
        P(m==2?f+k:f*(f+k)+k),R(m==2?P:2*P+A),env(true),model(start(env)),
        sep(R),vars(M*P),row(R,0),active(M*P,0){
        model.set(GRB_IntParam_Threads,1);model.set(GRB_IntParam_Seed,20260905);
        model.set(GRB_IntParam_Method,method);model.set(GRB_IntParam_DualReductions,0);
        model.set(GRB_DoubleParam_FeasibilityTol,1e-9);model.set(GRB_DoubleParam_OptimalityTol,1e-9);
        for(int q=0;q<M;++q)norm.push_back(model.addConstr(GRBLinExpr()==1.));
    }
    static GRBEnv& start(GRBEnv& e){e.set(GRB_IntParam_OutputFlag,0);e.start();return e;}
    int root(int s)const{return s<F*A?s/A:F+s-F*A;}
    int degree(int q)const{return M==2||q==0||q==3?1:2;}
    int edge(int q,int s,int slot)const{
        if(M==2||q==0)return s;
        if(q==1)return slot==0?s:P+root(s);
        if(q==2)return slot==0?P+root(s):P+A+s;
        return P+A+s;
    }
    double sign(int q,int slot)const{
        if(M==2)return q==0?1.:-1.;
        return q==0?1.:(q==3?-1.:(slot==0?-1.:1.));
    }
    void update(int count,const int* ids,const double* costs){
        for(int j=0;j<count;++j){int id=ids[j];
            if(id<0||id>=M*P||!std::isfinite(costs[j])||costs[j]<-1e-9)throw std::runtime_error("Invalid native column");
            int q=id/P,s=id%P;
            for(int slot=0;slot<degree(q);++slot){int e=edge(q,s,slot);
                if(!row[e]){sep[e]=model.addConstr(GRBLinExpr()==0.);row[e]=1;}}
        }model.update();
        for(int j=0;j<count;++j){int id=ids[j],q=id/P,s=id%P;
            if(active[id])vars[id].set(GRB_DoubleAttr_Obj,costs[j]);
            else{
                GRBColumn col;col.addTerm(1.,norm[q]);
                for(int slot=0;slot<degree(q);++slot)col.addTerm(sign(q,slot),sep[edge(q,s,slot)]);
                vars[id]=model.addVar(0.,GRB_INFINITY,costs[j],GRB_CONTINUOUS,col);
                active[id]=1;indices.push_back(id);
            }
        }model.update();
    }
    const char* solve(double seconds){
        model.set(GRB_DoubleParam_TimeLimit,std::max(1e-6,seconds));model.optimize();
        std::ostringstream os;os<<std::setprecision(17);int code=model.get(GRB_IntAttr_Status);
        os<<"{\"status\":"<<code<<",\"columns\":"<<indices.size()<<",\"rows\":"<<model.get(GRB_IntAttr_NumConstrs);
        if(code==GRB_OPTIMAL){
            double residual=0,minrc=1e100;
            os<<",\"objective\":"<<model.get(GRB_DoubleAttr_ObjVal)<<",\"alpha\":[";
            for(int q=0;q<M;++q){if(q)os<<",";os<<norm[q].get(GRB_DoubleAttr_Pi);
                residual=std::max(residual,std::abs(norm[q].get(GRB_DoubleAttr_Slack)));}
            os<<"],\"pi\":[";bool comma=false;
            for(int e=0;e<R;++e)if(row[e]){
                residual=std::max(residual,std::abs(sep[e].get(GRB_DoubleAttr_Slack)));
                double value=sep[e].get(GRB_DoubleAttr_Pi);if(value==0.)continue;
                if(comma)os<<",";comma=true;os<<"["<<e<<","<<value<<"]";
            }os<<"],\"support\":[";comma=false;
            for(int id:indices){minrc=std::min(minrc,vars[id].get(GRB_DoubleAttr_RC));
                double x=vars[id].get(GRB_DoubleAttr_X);if(x<=1e-9)continue;
                if(comma)os<<",";comma=true;os<<"["<<id<<","<<x<<"]";
            }
            os<<"],\"max_equality_residual\":"<<residual<<",\"minimum_active_reduced_cost\":"<<minrc;
            if(residual>1e-7||minrc<-1e-7)throw std::runtime_error("Native RMP numerical audit failed");
        }os<<"}";output=os.str();return output.c_str();
    }
};
}
API const char* crmp_error(){return error.c_str();}
API void* crmp_create(int M,int F,int K,int method){
    try{error.clear();if((M!=2&&M!=4)||F<1||K<1||4LL*(1LL*F*(F+1LL*K)+K)>2000000000LL)
        throw std::runtime_error("Invalid RMP dimensions");return new Master(M,F,K,method);
    }catch(const GRBException& e){error=e.getMessage();return nullptr;}catch(const std::exception& e){error=e.what();return nullptr;}
}
API int crmp_update(void* p,int n,const int* ids,const double* costs){
    try{static_cast<Master*>(p)->update(n,ids,costs);return 0;}
    catch(const GRBException& e){error=e.getMessage();return 1;}catch(const std::exception& e){error=e.what();return 1;}
}
API const char* crmp_solve(void* p,double seconds){
    try{return static_cast<Master*>(p)->solve(seconds);}
    catch(const GRBException& e){error=e.getMessage();return nullptr;}catch(const std::exception& e){error=e.what();return nullptr;}
}
API void crmp_destroy(void* p){delete static_cast<Master*>(p);}
