// Fixed-size FP64 actuator projection and owned spatial-hash lookup.
// Deliberately built without fast math or fused multiply-add contraction.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#define LUNAR_HD __host__ __device__
#include "upstream_native.h"

__global__ void project_kernel(const double *d,const double *p,double *out,int64_t n,Vehicle v) {
    int64_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    double x,y;
    bool ok=project(d[2*i],d[2*i+1],p[2*i],p[2*i+1],v,x,y);
    out[2*i]=ok?x:NAN;out[2*i+1]=ok?y:NAN;
}

extern "C" int project_cuda(const double *d,const double *p,double *out,int64_t n,const double *params,void *stream) {
    Vehicle v=*reinterpret_cast<const Vehicle*>(params);
    if(n)project_kernel<<<(n+127)/128,128,0,(cudaStream_t)stream>>>(d,p,out,n,v);
    return (int)cudaGetLastError();
}

__device__ void lookup_one(double qx,double qy,int64_t batch,
    const int64_t *keys,const int64_t *first,const int64_t *count,int64_t buckets,
    const double *xy,const double *features,int64_t ox,int64_t oy,int64_t nx,int64_t ny,
    double cell,double support,double *out,bool *known,bool *ambiguous,bool *boundary) {
    double sx=floor(qx/cell),sy=floor(qy/cell);
    double nearest=INFINITY;int64_t pick=0;
    int64_t slots[9];int found=0;
    if(isfinite(sx)&&isfinite(sy)&&fabs(sx)<=1099511627776.&&fabs(sy)<=1099511627776.) {
        int64_t cx=(int64_t)sx-ox,cy=(int64_t)sy-oy;
        for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy) {
            int64_t x=cx+dx,y=cy+dy;
            if(x<0||x>=nx||y<0||y>=ny)continue;
            int64_t key=(batch*nx+x)*ny+y,lo=0,hi=buckets;
            while(lo<hi){int64_t mid=(lo+hi)/2;if(keys[mid]<key)lo=mid+1;else hi=mid;}
            if(lo==buckets||keys[lo]!=key)continue;
            slots[found++]=lo;
            for(int64_t j=first[lo];j<first[lo]+count[lo];++j) {
                double x=xy[2*j]-qx,y=xy[2*j+1]-qy,d=sqrt(x*x+y*y);
                if(d<nearest){nearest=d;pick=j;}
            }
        }
    }
    bool valid=nearest<=support+1e-10;int ties=0;
    if(valid)for(int s=0;s<found;++s) {
        int64_t slot=slots[s];
        for(int64_t j=first[slot];j<first[slot]+count[slot];++j) {
            double x=xy[2*j]-qx,y=xy[2*j+1]-qy,d=sqrt(x*x+y*y);
            ties+=fabs(d-nearest)<=1e-8+2e-12;
        }
    }
    for(int c=0;c<5;++c)out[c]=valid?features[5*pick+c]:0.;
    *known=valid;*ambiguous=valid&&ties>1;*boundary=isfinite(nearest)&&fabs(nearest-support)<=2e-10;
}

__global__ void lookup_kernel(const double *q,int64_t n,int64_t per_env,
    const int64_t *keys,const int64_t *first,const int64_t *count,int64_t buckets,
    const double *xy,const double *features,int64_t ox,int64_t oy,int64_t nx,int64_t ny,
    double cell,double support,double *out,bool *known,bool *ambiguous,bool *boundary) {
    int64_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    lookup_one(q[2*i],q[2*i+1],i/per_env,keys,first,count,buckets,xy,features,ox,oy,nx,ny,
               cell,support,out+5*i,known+i,ambiguous+i,boundary+i);
}

extern "C" int lookup_cuda(const double *q,int64_t n,int64_t per_env,
    const int64_t *keys,const int64_t *first,const int64_t *count,int64_t buckets,
    const double *xy,const double *features,int64_t ox,int64_t oy,int64_t nx,int64_t ny,
    double cell,double support,double *out,bool *known,bool *ambiguous,bool *boundary,void *stream) {
    if(n)lookup_kernel<<<(n+127)/128,128,0,(cudaStream_t)stream>>>(q,n,per_env,keys,first,count,buckets,xy,features,
        ox,oy,nx,ny,cell,support,out,known,ambiguous,boundary);
    return (int)cudaGetLastError();
}

__global__ void rollout_kernel(const double *raw,const double *pose,const double *previous,const double *caps,
    int64_t n,int64_t per_env,Vehicle v,double dt,double slope_scale,
    const int64_t *keys,const int64_t *first,const int64_t *count,int64_t buckets,
    const double *xy,const double *features,int64_t ox,int64_t oy,int64_t nx,int64_t ny,
    double cell,double support,double *states,double *controls,double *velocity,bool *numeric,
    bool *known,bool *ambiguous,bool *boundary,double *sampled) {
    int64_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    int64_t b=i/per_env;
    double x=pose[3*b],y=pose[3*b+1],yaw=pose[3*b+2],pv=previous[2*b],pw=previous[2*b+1];
    bool stopping=false,valid=true;
    for(int t=0;t<=40;++t) {
        int64_t node=i*41+t;
        states[3*node]=x;states[3*node+1]=y;states[3*node+2]=yaw;
        lookup_one(x,y,b,keys,first,count,buckets,xy,features,ox,oy,nx,ny,cell,support,
                   sampled+5*node,known+node,ambiguous+node,boundary+node);
        if(t==40)break;
        int64_t control=2*(i*40+t);
        double dv=raw[control],dw=raw[control+1];
        valid=valid&&isfinite(dv)&&isfinite(dw);
        if(!isfinite(dv))dv=0.;if(!isfinite(dw))dw=0.;
        dv=fmax(-fmin(caps[b],v.reverse),fmin(caps[b],dv));
        double bound=ceil(fabs(pv)/v.decv-1e-10)+ceil(fabs(pw)/v.decw-1e-10)+1.;
        stopping=stopping||bound>=40-t;
        double cv,cw;
        if(!project(stopping?0.:dv,stopping?0.:dw,pv,pw,v,cv,cw)){cv=NAN;cw=NAN;}
        if(fabs(cv)<=1e-10)cv=0.;if(fabs(cw)<=1e-10)cw=0.;
        controls[control]=cv;controls[control+1]=cw;pv=cv;pw=cw;
        double *f=sampled+5*node;
        double along=f[1]*cos(yaw)+f[2]*sin(yaw);
        double gain=fmin(1.,fmax(.25,f[4]*exp(-fabs(along)*slope_scale)));
        double av=cv*gain,aw=cw*gain,mid=yaw+.5*dt*aw;
        x=x+dt*(av*cos(mid));y=y+dt*(av*sin(mid));yaw=yaw+dt*aw;
        velocity[control]=av;velocity[control+1]=aw;
    }
    numeric[i]=valid;
}

extern "C" int rollout_cuda(const double *raw,const double *pose,const double *previous,const double *caps,
    int64_t n,int64_t per_env,const double *params,double dt,double slope_scale,
    const int64_t *keys,const int64_t *first,const int64_t *count,int64_t buckets,
    const double *xy,const double *features,int64_t ox,int64_t oy,int64_t nx,int64_t ny,
    double cell,double support,double *states,double *controls,double *velocity,bool *numeric,
    bool *known,bool *ambiguous,bool *boundary,double *sampled,void *stream) {
    Vehicle v=*reinterpret_cast<const Vehicle*>(params);
    if(n)rollout_kernel<<<(n+127)/128,128,0,(cudaStream_t)stream>>>(raw,pose,previous,caps,n,per_env,v,dt,slope_scale,
        keys,first,count,buckets,xy,features,ox,oy,nx,ny,cell,support,states,controls,velocity,numeric,
        known,ambiguous,boundary,sampled);
    return (int)cudaGetLastError();
}
