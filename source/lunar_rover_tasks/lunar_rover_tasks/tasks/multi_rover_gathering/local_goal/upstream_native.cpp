#include <cstdint>
#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>
#include "upstream_native.h"
extern "C" int project_cpu(const double *d,const double *p,double *out,int64_t n,const double *params) {
    Vehicle v=*reinterpret_cast<const Vehicle*>(params);
    for(int64_t i=0;i<n;++i)if(!project(d[2*i],d[2*i+1],p[2*i],p[2*i+1],v,out[2*i],out[2*i+1],true))return 1;
    return 0;
}

extern "C" int brake_sequence_cpu(const double *previous,double *out,int64_t capacity,const double *params) {
    Vehicle v=*reinterpret_cast<const Vehicle*>(params);
    double pv=previous[0],pw=previous[1];
    for(int64_t i=0;i<capacity;++i) {
        double nv,nw;
        if(!project(0.,0.,pv,pw,v,nv,nw,true))return -1;
        if(std::abs(nv)<1e-10)nv=0.;
        if(std::abs(nw)<1e-10)nw=0.;
        out[2*i]=nv;out[2*i+1]=nw;pv=nv;pw=nw;
        if(nv==0. && nw==0.)return int(i+1);
    }
    return 0;
}

extern "C" int control_error_cpu(const double *u,const double *p,const double *params,double radius,double tolerance) {
    Vehicle v=*reinterpret_cast<const Vehicle*>(params);
    if(u[0]<-v.reverse-tolerance||u[0]>v.forward+tolerance)return 1;
    if(fabs(u[1])>(fabs(u[0])<=v.low?v.lowyaw:v.yaw)+tolerance)return 2;
    double left=(u[0]-v.track*u[1])/radius,right=(u[0]+v.track*u[1])/radius;
    if(fmax(fabs(left),fabs(right))>v.wheel/radius+tolerance)return 3;
    double acc[2]={v.accv,v.accw},dec[2]={v.decv,v.decw};
    for(int j=0;j<2;++j) {
        double lo=p[j]>0?fmax(0.,p[j]-dec[j]):p[j]-acc[j];
        double hi=p[j]<0?fmin(0.,p[j]+dec[j]):p[j]+acc[j];
        if(u[j]<lo-tolerance||u[j]>hi+tolerance)return 4;
    }
    if(fmax(fabs(left-(p[0]-v.track*p[1])/radius),fabs(right-(p[0]+v.track*p[1])/radius))>v.step/radius+tolerance)return 5;
    return 0;
}

extern "C" int sequence_valid_cpu(const double *controls,const double *previous,int64_t n,const double *params,double radius) {
    for(int64_t i=0;i<n;++i)if(control_error_cpu(controls+2*i,i?controls+2*(i-1):previous,params,radius,1e-8))return 0;
    return 1;
}

namespace {
struct PointKey {
    double x,y;
    bool operator==(const PointKey &b) const {return x==b.x && y==b.y;}
};
struct CellKey {
    int64_t x,y;
    bool operator==(const CellKey &b) const {return x==b.x && y==b.y;}
};
struct PointHash {
    size_t operator()(const PointKey &p) const {
        // std::hash<double> preserves Python's equality of +0 and -0.
        return std::hash<double>{}(p.x) ^ (std::hash<double>{}(p.y)<<1);
    }
};
struct CellHash {
    size_t operator()(const CellKey &p) const {
        return std::hash<int64_t>{}(p.x) ^ (std::hash<int64_t>{}(p.y)<<1);
    }
};
struct Observation {PointKey p; std::array<double,5> f; double t;};
double slope(const double *f) {return std::sqrt(f[1]*f[1]+f[2]*f[2]);}
bool boundary(double a,double b) {return std::abs(a-b)<=1e-12*(1.+std::abs(a)+std::abs(b));}
void merge_feature(const double *a,double *b,double step,bool &ambiguous) {
    double na=slope(a),nb=slope(b);
    if ((a[1]!=b[1] || a[2]!=b[2]) && boundary(na,nb)) ambiguous=true;
    bool hole=std::abs(a[0]-b[0])>step+1e-8;
    if(boundary(std::abs(a[0]-b[0]),step+1e-8)) ambiguous=true;
    b[0]=std::max(a[0],b[0]); b[3]=std::max(a[3],b[3]); b[4]=std::min(a[4],b[4]);
    if(na>nb) {b[1]=a[1];b[2]=a[2];}
    if(hole)b[4]=0.;
}
}

// Stable direct-point merge. Hash tables only look up identities; output order
// is explicitly old survivors followed by first-insertion-ordered fresh keys.
// Numerical decision boundaries return 2 for the scalar NumPy authority.
extern "C" int memory_merge_cpu(const double *oldxy,const double *oldf,const double *oldt,
    const int64_t *oldcells,int64_t n,const double *xy,const double *features,
    const int64_t *cells,int64_t m,double timestamp,int64_t capacity,const double *limits,
    double *outxy,double *outf,double *outt,int64_t *counts) {
    // Per-native-thread scratch never aliases an exported immutable view.
    struct Chain {int64_t head=-1,tail=-1;};
    thread_local std::unordered_map<PointKey,int64_t,PointHash> keys;
    thread_local std::unordered_map<CellKey,Chain,CellHash> by_cell;
    thread_local std::vector<Observation> fresh, observations;
    thread_local std::vector<int64_t> next, survivors;
    keys.clear();by_cell.clear();fresh.clear();observations.clear();survivors.clear();
    keys.reserve(m);by_cell.reserve(m);fresh.reserve(m);observations.reserve(m);
    next.assign(m,-1);survivors.reserve(n);
    bool ambiguous=false;
    for(int64_t i=0;i<m;++i) {
        Observation value{{xy[2*i],xy[2*i+1]}, {},timestamp};
        std::copy_n(features+5*i,5,value.f.begin());
        auto found=keys.find(value.p);
        if(found!=keys.end()) {
            merge_feature(fresh[found->second].f.data(),value.f.data(),limits[3],ambiguous);
            fresh[found->second]=value;
        } else {keys.emplace(value.p,fresh.size());fresh.push_back(value);}
        auto &chain=by_cell[{cells[2*i],cells[2*i+1]}];
        if(chain.tail>=0)next[chain.tail]=i;else chain.head=i;
        chain.tail=i;observations.push_back(value);
    }
    if(int64_t(fresh.size())>capacity)return 1;
    for(int64_t i=0;i<n;++i) {
        auto cell=by_cell.find({oldcells[2*i],oldcells[2*i+1]});
        const double *f=oldf+5*i;bool keep=cell==by_cell.end();
        if(!keep) {
            double norm=slope(f);
            if(boundary(norm,limits[1]))ambiguous=true;
            keep=f[4]<limits[0] || norm>limits[1] || f[3]>limits[2];
            if(!keep)for(int64_t k=cell->second.head;k>=0;k=next[k]) {
                const auto &v=observations[k];
                double delta=f[0]-v.f[0]-(v.f[1]*(oldxy[2*i]-v.p.x)+v.f[2]*(oldxy[2*i+1]-v.p.y));
                if(boundary(std::abs(delta),limits[3]+1e-8))ambiguous=true;
                if(std::abs(delta)>limits[3]+1e-8){keep=true;break;}
            }
        }
        if(!keep)continue;
        auto replacement=keys.find({oldxy[2*i],oldxy[2*i+1]});
        if(replacement!=keys.end())merge_feature(f,fresh[replacement->second].f.data(),limits[3],ambiguous);
        else survivors.push_back(i);
    }
    if(ambiguous)return 2;
    int64_t total=survivors.size()+fresh.size(),drop=std::max(int64_t(0),total-capacity),out=0;
    for(int64_t j=drop;j<int64_t(survivors.size());++j) {
        int64_t i=survivors[j];outxy[2*out]=oldxy[2*i];outxy[2*out+1]=oldxy[2*i+1];
        std::copy_n(oldf+5*i,5,outf+5*out);outt[out++]=oldt[i];
    }
    for(const auto &v:fresh) {
        outxy[2*out]=v.p.x;outxy[2*out+1]=v.p.y;
        std::copy(v.f.begin(),v.f.end(),outf+5*out);outt[out++]=timestamp;
    }
    counts[0]=out;counts[1]=fresh.size();return 0;
}
