#pragma once
#include <cmath>
using std::isfinite;
#ifndef LUNAR_HD
#define LUNAR_HD
#endif
struct Vehicle {
    double accv, accw, decv, decw, reverse, forward, yaw, low, lowyaw, track, wheel, step;
};

LUNAR_HD bool project(double dv, double dw, double pv, double pw,
                               Vehicle v, double &ov, double &ow, bool numpy_order=false) {
    const double ax[12]={v.accv,0,-v.accv,0,v.accv,v.accv,-v.accv,-v.accv,v.accv,v.accv,-v.accv,-v.accv};
    const double ay[12]={0,v.accw,0,-v.accw,-v.track*v.accw,v.track*v.accw,v.track*v.accw,-v.track*v.accw,
                        -v.track*v.accw,v.track*v.accw,v.track*v.accw,-v.track*v.accw};
    if (!isfinite(dv)||!isfinite(dw)||!isfinite(pv)||!isfinite(pw)) return false;
    if (!numpy_order && (pv < -v.reverse-1e-10 || pv>v.forward+1e-10
        || fabs(pw)>(fabs(pv)<=v.low?v.lowyaw:v.yaw)+1e-10
        || fmax(fabs(pv-v.track*pw),fabs(pv+v.track*pw))>v.wheel+1e-10)) return false;
    double tx=fmin(fmax(dv,-v.reverse),v.forward)/v.accv;
    double ty=fmin(fmax(dw,-v.yaw),v.yaw)/v.accw;
    double vl=fmax(pv>0?fmax(0.,pv-v.decv):pv-v.accv,-v.reverse);
    double vh=fmin(pv<0?fmin(0.,pv+v.decv):pv+v.accv,v.forward);
    double wl=fmax(pw>0?fmax(0.,pw-v.decw):pw-v.accw,-v.yaw);
    double wh=fmin(pw<0?fmin(0.,pw+v.decw):pw+v.accw,v.yaw);
    double best=INFINITY;
    for(int mode=0;mode<3;++mode) {
        double ml=mode==0?-v.low:mode==1?v.low+1e-9:-v.reverse;
        double mh=mode==0?v.low:mode==1?v.forward:-v.low-1e-9;
        double cap=mode==0?v.lowyaw:v.yaw;
        double b[12]={fmin(vh,mh),fmin(wh,cap),-fmax(vl,ml),-fmax(wl,-cap),
                     v.wheel,v.wheel,v.wheel,v.wheel,
                     pv-v.track*pw+v.step,pv+v.track*pw+v.step,
                     -(pv-v.track*pw)+v.step,-(pv+v.track*pw)+v.step};
        if (-b[2]>b[0]+2e-10 || -b[3]>b[1]+2e-10) continue;
        // Match Torch's target, edge, then lexicographic upper-triangle order.
        for(int candidate=0;candidate<79;++candidate) {
            double x=tx,y=ty;
            int edge=candidate-1, pi=-1, pj=-1;
            if(numpy_order && candidate>0) {
                int k=candidate-1,i=0;
                while(k>=i+1){k-=i+1;++i;}
                if(k==0)edge=i;
                else {edge=-1;pi=k-1;pj=i;}
            } else if(candidate>12) {
                int k=candidate-13,i=0;
                while(k>=11-i){k-=11-i;++i;}
                edge=-1;pi=i;pj=i+1+k;
            }
            if(candidate>0 && edge>=0) {
                int i=edge;
                double d=((tx*ax[i]+ty*ay[i])-b[i])/(ax[i]*ax[i]+ay[i]*ay[i]);
                x=tx-ax[i]*d; y=ty-ay[i]*d;
            } else if(candidate>0) {
                int i=pi,j=pj;
                double det=ax[i]*ay[j]-ay[i]*ax[j];
                if(fabs(det)<=1e-15)continue;
                x=(b[i]*ay[j]-b[j]*ay[i])/det;
                y=(ax[i]*b[j]-ax[j]*b[i])/det;
            }
            double cost=(x-tx)*(x-tx)+(y-ty)*(y-ty);
            if(cost>=best)continue;
            bool feasible=true;
            for(int i=0;i<12;++i)if(x*ax[i]+y*ay[i]>b[i]+1e-10){feasible=false;break;}
            if(feasible){best=cost;ov=x*v.accv;ow=y*v.accw;if(cost==0.)return true;}
        }
    }
    return isfinite(best);
}

