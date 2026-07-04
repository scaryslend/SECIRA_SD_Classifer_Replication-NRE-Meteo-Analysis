/*
 Oleg I.Berngardt (2025)
*/
#include "classifierV32.h"
#include "classifierV32_coefs0.h"
#include "classifierV32_coefs1.h"
#include "classifierV32_coefs2.h"
#define LAYER1_SIZE 56
#define LAYER2_SIZE 37

int argmax(float* arr, float* prob, int width_out_)
 {
  int pos=-1;
  float max=-1;
  if (prob)
   *prob=-1;
  for(int i=0;i<width_out_;i++)
   if (arr[i]>max || i==0)
    { max=arr[i]; pos=i; }
  if (prob)
   {
//   printf("found: %d %e %e\n",pos,max,arr[pos]);
   *prob=(float)max;
   }
  return pos;
 }



/// Class predictor, V3.2
/// Trained at 12 SuperDARN and SECIRA radars, for low and high solar activity
int predictV32_0(float* inp, float*outp)
/// inp:
/// V,Wl,Hiri,Mode,sin(k,xy)[R],cos(k,B)[R],sin(k,xy)[R/4],sin(k,xy)[R/2],sin(k,xy)[3R/4]
/// outp: class probabilities (Kolmogorov-like)
{
 double x[100];
 double y[100];
 for(int i=0;i<100;i++)
  x[i]=y[i]=0.;

/*
 for(int i=0;i<9;i++)
  {
   printf("%.15e ",inp[i]);
   if(i%5==4) printf("\n");
  }
 exit(1);
*/
/// batch normalization
 for(int i=0;i<9;i++)
  x[i]=(inp[i]-weights_bn_2_0[i])/sqrt(weights_bn_3_0[i]+1e-3);
 for(int i=0;i<9;i++)
   x[i]=x[i]*weights_bn_0_0[i]+weights_bn_1_0[i];
/*

 for(int i=0;i<9;i++)
  {
   printf("%.15f\n",x[i]);
//   if(i%5==4) printf("\n");
  }
 exit(1);
*/ 
/// transform1
 for(int i=0;i<LAYER1_SIZE;i++)
  {
   y[i]=weights_tr1_1_0[i];
   for(int j=0;j<9;j++)
    y[i]+=weights_tr1_0_0[j][i]*x[j];
  }
 for(int i=0;i<LAYER1_SIZE;i++)
  x[i]=fabs(y[i]);

/*
 for(int i=0;i<LAYER1_SIZE;i++)
  {
   printf("%.10e ",x[i]);
   if(i%5==4) printf("\n");
  }
 exit(1);
*/


/// transform2
 for(int i=0;i<LAYER2_SIZE;i++)
  {
   y[i]=weights_tr2_1_0[i];
   for(int j=0;j<LAYER1_SIZE;j++)
    y[i]+=weights_tr2_0_0[j][i]*x[j];
  }
 for(int i=0;i<LAYER2_SIZE;i++)
   x[i]=fabs(y[i]);
/*
 for(int i=0;i<LAYER2_SIZE;i++)
  {
   printf("%f ",x[i]);
   if(i%5==4) printf("\n");
  }
 exit(1);
*/

/// norm layer
 float sum=0;
 for(int i=0;i<LAYER2_SIZE;i++)
  sum+=x[i];
 for(int i=0;i<LAYER2_SIZE;i++)
  outp[i]=x[i]/sum;
 float prob;
 return argmax(outp,&prob,LAYER2_SIZE);
}

int predictV32_1(float* inp, float*outp)
/// inp:
/// V,Wl,Hiri,Mode,sin(k,xy)[R],cos(k,B)[R],sin(k,xy)[R/4],sin(k,xy)[R/2],sin(k,xy)[3R/4]
/// outp: class probabilities (Kolmogorov-like)
{
 float x[100];
 float y[100];
 for(int i=0;i<100;i++)
  x[i]=y[i]=0.;

/// batch normalization
 for(int i=0;i<9;i++)
  x[i]=(inp[i]-weights_bn_2_1[i])/sqrt(weights_bn_3_1[i]+1e-3);
 for(int i=0;i<9;i++)
   x[i]=x[i]*weights_bn_0_1[i]+weights_bn_1_1[i];

/// transform1
 for(int i=0;i<LAYER1_SIZE;i++)
  {
   y[i]=weights_tr1_1_1[i];
   for(int j=0;j<9;j++)
    y[i]+=weights_tr1_0_1[j][i]*x[j];
  }
 for(int i=0;i<LAYER1_SIZE;i++)
  x[i]=fabs(y[i]);

/// transform2
 for(int i=0;i<LAYER2_SIZE;i++)
  {
   y[i]=weights_tr2_1_1[i];
   for(int j=0;j<LAYER1_SIZE;j++)
    y[i]+=weights_tr2_0_1[j][i]*x[j];
  }
 for(int i=0;i<LAYER2_SIZE;i++)
   x[i]=fabs(y[i]);

/// norm layer
 float sum=0;
 for(int i=0;i<LAYER2_SIZE;i++)
  sum+=x[i];
 for(int i=0;i<LAYER2_SIZE;i++)
  outp[i]=x[i]/sum;
 float prob;
 return argmax(outp,&prob,LAYER2_SIZE);
}

int predictV32_2(float* inp, float*outp)
/// inp:
/// V,Wl,Hiri,Mode,sin(k,xy)[R],cos(k,B)[R],sin(k,xy)[R/4],sin(k,xy)[R/2],sin(k,xy)[3R/4]
/// outp: class probabilities (Kolmogorov-like)
{
 float x[100];
 float y[100];
 for(int i=0;i<100;i++)
  x[i]=y[i]=0.;

/// batch normalization
 for(int i=0;i<9;i++)
  x[i]=(inp[i]-weights_bn_2_2[i])/sqrt(weights_bn_3_2[i]+1e-3);
 for(int i=0;i<9;i++)
   x[i]=x[i]*weights_bn_0_2[i]+weights_bn_1_2[i];

/// transform1
 for(int i=0;i<LAYER1_SIZE;i++)
  {
   y[i]=weights_tr1_1_2[i];
   for(int j=0;j<9;j++)
    y[i]+=weights_tr1_0_2[j][i]*x[j];
  }
 for(int i=0;i<LAYER1_SIZE;i++)
  x[i]=fabs(y[i]);

/// transform2
 for(int i=0;i<LAYER2_SIZE;i++)
  {
   y[i]=weights_tr2_1_2[i];
   for(int j=0;j<LAYER1_SIZE;j++)
    y[i]+=weights_tr2_0_2[j][i]*x[j];
  }
 for(int i=0;i<LAYER2_SIZE;i++)
   x[i]=fabs(y[i]);

/// norm layer
 float sum=0;
 for(int i=0;i<LAYER2_SIZE;i++)
  sum+=x[i];
 for(int i=0;i<LAYER2_SIZE;i++)
  outp[i]=x[i]/sum;
 float prob;
 return argmax(outp,&prob,LAYER2_SIZE);
}





// ensemble realization
int predictV32(float* inp, float*outp)
{
 int res0=predictV32_0(inp, outp);
// return res0;
 int res1=predictV32_1(inp, outp);
 int res2=predictV32_2(inp, outp);
 if (res0==res1 && res0==res2)
  return res0;
 else
  return LAYER2_SIZE;
}


#ifdef __TEST__
#include <stdio.h>
int main()
{
/*
 float x=0.3090000000001;
 printf("%.15f",x);
 exit(1);
*/
  float inp[9]={0,0,0,0,0,0,0,0,0};
// float inp[9]={1,1,1,1,1,1,1,1,1};
 float outp[LAYER2_SIZE];
 float pr;
 int pos;
 int i,k;
// for (k=0;k<1 && !feof(stdin);k++)
 for (k=0;!feof(stdin);k++)
 {
  int res;
  for(i=0;i<9;i++)
  {
   res=fscanf(stdin,"%e",&(inp[i]));
   if (res!=1) break;
  }
  if (res!=1) break;

  predictV32(inp,outp);
/// display
  pos=argmax(outp,&pr,LAYER2_SIZE);
//  printf("prob: %e pos: %d, v:%le\n",pr,pos,outp[pos]);
/*
  for(int i=0;i<LAYER2_SIZE;i++)
//   printf("%e ",outp[i]/pr);
   {
//    if(i%5==0)
//     printf("\n");
    printf("%.15f\n",outp[i]);
   }
//  printf("\n class: %d\n",pos);
*/
  printf("%d\n",pos);
 }
}

#endif

