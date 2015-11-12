#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <getopt.h>

#define LIM 128

static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex a,float s)
{
  cufftComplex c;
  c.x=s*a.x;
  c.y=s*a.y;
  return c;
}

static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a,cufftComplex b)
{
  cufftComplex c;
  c.x=a.x*b.x-a.y*b.y;
  c.y=a.x*b.y+a.y*b.x;
  return c;
}

static __global__ void PointwiseComplexMultiply(cufftComplex *a,cufftComplex *b,cufftComplex *c,int n,int m,int j,float scale)
{
  int i;
  const int numThreads=blockDim.x*gridDim.x;
  const int threadID=blockIdx.x*blockDim.x+threadIdx.x;

  for (i=threadID;i<n;i+=numThreads)
    c[i]=ComplexScale(ComplexMul(a[i],b[i%m+j*m]),scale);
}

__global__ void padd_data(float *y,float *ytmp,int nsamp,int nbin,int m,int nfft)
{
  int isamp,idx,ibin,ifft,ioverlap;

  // Indices of input data                                                                    
  ibin=blockIdx.x*blockDim.x+threadIdx.x;
  ifft=blockIdx.y*blockDim.y+threadIdx.y;

  // Compute valid threads
  if (ibin<nbin && ifft<nfft) {
    ioverlap=(nbin-m)/2;
    
    for (ifft=0;ifft<nfft;ifft++) {
      idx=ibin+nbin*ifft;
      if (idx<nbin*nfft) {
	isamp=ibin+m*ifft-ioverlap;
	if (isamp<0 || isamp>=nsamp)
	  ytmp[idx]=0.0;
	else
	  ytmp[idx]=y[isamp];
      }
    }
  }
  
  return;
}

__global__ void unpadd_data(float *y,float *ytmp,int n,int nx,int m,int ny)
{
  int i,j,k,l;
  int ioverlap;
  const int numThreads=blockDim.x*gridDim.x;
  const int threadID=blockIdx.x*blockDim.x+threadIdx.x;

  for (i=threadID;i<nx;i+=numThreads) {
    ioverlap=(nx-m)/2;
    
    for (j=0;j<ny;j++) {
      k=i+nx*j;
      if (k<nx*ny) {
	l=i+m*j-ioverlap;
	if (l>=0 && l<n && i>=ioverlap && i<m+ioverlap)
	  y[l]=ytmp[k];
      }
    }
  }

  return;
}

__global__ void prune(float *z,int n,int dw,int *mask,float sigma)
{
  int i,j;
  const int numThreads=blockDim.x*gridDim.x;
  const int threadID=blockIdx.x*blockDim.x+threadIdx.x;

  for (i=threadID;i<n;i+=numThreads) {
    mask[i]=1;
    if (z[i]<sigma) 
      mask[i]=0;
    for (j=i-dw/2;j<=i+dw/2;j++) {
      if (j<0 || j>=n)
	continue;
      if (z[j]<sigma)
	continue;
      if (z[j]>z[i])
	mask[i]=0;
    }
  }
  return;
}

__global__ void prune_final(float *z,int *dw,int *mask,int n,float sigma)
{
  int i,j;
  const int numThreads=blockDim.x*gridDim.x;
  const int threadID=blockIdx.x*blockDim.x+threadIdx.x;

  for (i=threadID;i<n;i+=numThreads) {
    mask[i]=1;

    // Mask candidates within half-width with lower significance
    if (z[i]>sigma) {
      for (j=i-dw[i]/2;j<=i+dw[i]/2;j++) {
	if (j<0 || j>=n)
	  continue;
	if (z[j]<z[i])
	  mask[j]=0;
      }
    }
  }

  return;
}

__global__ void store(float *x,float *z,int *mask,int *w,int wnew,int n)
{
  int i;
  const int numThreads=blockDim.x*gridDim.x;
  const int threadID=blockIdx.x*blockDim.x+threadIdx.x;

  for (i=threadID;i<n;i+=numThreads) {
    if (z[i]*mask[i]>x[i]) {
      x[i]=z[i]*mask[i];
      w[i]=wnew;
    }
  }

  return;
}

__global__ void store_final(float *x,int *mask,int n,float sigma)
{
  int i;
  const int numThreads=blockDim.x*gridDim.x;
  const int threadID=blockIdx.x*blockDim.x+threadIdx.x;

  for (i=threadID;i<n;i+=numThreads) {
    x[i]*=mask[i];
    if (x[i]<sigma) 
      x[i]=0.0;
  }

  return;
}

__global__ void detrend_and_normalize(float *y,float *ytmp,int nsamp,int mdetrend,int ndetrend,int ndm)
{
  int j,l,isamp,isamp0,isamp1;
  int idetrend,idm,idx;
  float x,s,sx,sxx,sy,sxy,syy,d,a,b;
  int isampmin,isampmax,lmax;
  float ymin,ymax,yswap,ystd;

  idetrend=blockIdx.x*blockDim.x+threadIdx.x;
  idm=blockIdx.y*blockDim.y+threadIdx.y;
  if (idetrend>=ndetrend || idm>=ndm)
    return;

  // Compute sums
  s=sx=sxx=sy=sxy=0.0;
  for (j=0;j<mdetrend;j++) {
    isamp=idetrend*mdetrend+j;
    idx=isamp+idm*nsamp;
    if (isamp>=nsamp)
      break;
    x=-0.5+(float) j/(float) mdetrend;
    s+=1.0;
    sx+=x;
    sxx+=x*x;
    sy+=y[idx];
    sxy+=x*y[idx];
  }

  // Linear parameters
  d=s*sxx-sx*sx;
  a=(sxx*sy-sx*sxy)/d;
  b=(s*sxy-sx*sy)/d;

  // Remove trend
  s=syy=0.0;
  for (j=0;j<mdetrend;j++) {
    isamp=idetrend*mdetrend+j;
    idx=isamp+idm*nsamp;
    if (isamp>=nsamp)
      break;
    x=-0.5+(float) j/(float) mdetrend;
    y[idx]-=a+b*x;
    ytmp[idx]=y[idx];
    s+=1.0;
    syy+=y[idx]*y[idx];
  }

  // Remove outliers 2.5% on either end
  isamp0=idetrend*mdetrend;
  isamp1=(idetrend+1)*mdetrend;
  lmax=mdetrend;
  if (isamp1>=nsamp) {
    lmax=nsamp-idetrend*mdetrend;
    isamp1=nsamp;
  }
  for (l=0;l<0.025*lmax;l++) {
    for (j=l;j<lmax-l;j++) {
      isamp=isamp0+j;
      if (isamp>=nsamp)
        break;
      if (j==l || ytmp[isamp+idm*nsamp]<ymin) {
        ymin=ytmp[isamp+idm*nsamp];
        isampmin=isamp;
      }
      if (j==l || ytmp[isamp+idm*nsamp]>ymax) {
        ymax=ytmp[isamp+idm*nsamp];
        isampmax=isamp;
      }
    }

    yswap=ytmp[isamp0+l+idm*nsamp];
    ytmp[isamp0+l+idm*nsamp]=ytmp[isampmin+idm*nsamp];
    ytmp[isampmin+idm*nsamp]=yswap;
    yswap=ytmp[isamp1-l-1+idm*nsamp];
    ytmp[isamp1-l-1+idm*nsamp]=ytmp[isampmax+idm*nsamp];
    ytmp[isampmax+idm*nsamp]=yswap;

    // Adjust sum
    syy-=ymin*ymin+ymax*ymax;
    s-=2.0;
  }
  ystd=1.148*sqrt(syy/s);

  // Normalize
  for (j=0;j<mdetrend;j++) {
    isamp=idetrend*mdetrend+j;
    if (isamp>=nsamp)
      break;
    y[isamp+idm*nsamp]/=ystd;
  }

  return;
}

void usage(void)
{

  printf("usage:  single_pulse_search.py [options] .dat files _or_ .singlepulse files\n");
  printf(" [-h  ]    : Display this help\n");
  printf(" [-m  ]    : Set the max downsampling in sec (see below for default)\n");
  printf(" [-t  ]    : Set a different threshold SNR (default=5.0)\n");

  return;
}

// Read a line of maximum length int lim from file FILE into string s
int fgetline(FILE *file,char *s,int lim)
{
  int c,i=0;

  while (--lim > 0 && (c=fgetc(file)) != EOF && c != '\n')
    s[i++] = c;
  if (c == '\n')
    s[i++] = c;
  s[i] = '\0';
  return i;
}

int read_info_file(char *fname,float *dt,float *dm,int *n)
{
  FILE *file;
  char line[LIM];
  int flag=0;

  // Open file
  file=fopen(fname,"r");
  if (file==NULL) {
    fprintf(stderr,"Error opening %s\n",fname);
    return -1;
  }

  // Loop over file contents
  while (fgetline(file,line,LIM)>0) {
    // Find sample time
    if (strstr(line,"Width of each time series bin (sec)")!=NULL) {
      sscanf(line+43,"%f",dt);
      flag++;
    }

    // Dispersion measure
    if (strstr(line,"Dispersion measure (cm-3 pc)")!=NULL) {
      sscanf(line+43,"%f",dm);
      flag++;
    }

    // Number of samples
    if (strstr(line,"Number of bins in the time series")!=NULL) {
      sscanf(line+43,"%d",n);
      flag++;
    }
  }
  
  // Close file
  fclose(file);

  // Error if information not found
  if (flag!=3) {
    fprintf(stderr,"Dispersion measure, sampling time or number of samples keywords not found in %s\n",fname);
    return -1;
  }

  return 0;
}

int main(int argc,char *argv[])
{
  int i,j,k,nx,mx,my,ny,nsamp=0,m,mdetrend,ndetrend,ndm=2;
  cufftHandle ftr2cx,ftr2cy,ftc2rz;
  float *x,*y,*z,*dxs,*dzs,*fbuf;
  cufftReal *dx,*dy,*dz;
  cufftComplex *dcx,*dcy,*dcz;
  int *dmask,*mask,*dw,*w;
  int idist,odist,iembed,oembed,istride,ostride;
  int ds[]={2,3,4,6,9,14,20,30,45,70,100,150},dsmax=30;
  FILE *file;
  float dt=0.0001,dm=0.0,sigma=5.0,wmax=0.0;
  char *datfname,*inffname,*spfname;
  int arg=0,len;
  dim3 gridsize,blocksize;

  // Decode options
  if (argc>1) {
    while ((arg=getopt(argc,argv,"hm:t:"))!=-1) {
      switch(arg) {
	
      case 't':
	sigma=atof(optarg);
	break;
	
      case 'm':
	wmax=atof(optarg);
	break;

      case 'h':
        usage();
        return 0;

      default:
        usage();
	return 0;
      }
    }
  } else {
    usage();
    return 0;
  }

  // Set filenames
  len=strlen(argv[optind]);
  datfname=(char *) malloc(sizeof(char)*(len+2));
  inffname=(char *) malloc(sizeof(char)*(len+2));
  spfname=(char *) malloc(sizeof(char)*(len+15));

  // Assuming timeseries filename ends in .dat
  strcpy(datfname,argv[optind]);
  argv[optind][len-4]='\0';
  sprintf(inffname,"%s.inf",argv[optind]);
  sprintf(spfname,"%s.singlepulse",argv[optind]);

  // Read inf file
  if (read_info_file(inffname,&dt,&dm,&nsamp)!=0)
    return -1;

  // Allocate signal timeseries
  x=(float *) malloc(sizeof(float)*nsamp*ndm);
  z=(float *) malloc(sizeof(float)*nsamp*ndm);
  fbuf=(float *) malloc(sizeof(float)*nsamp);

  // Open file
  file=fopen(datfname,"r");
  if (file==NULL) {
    fprintf(stderr,"Error opening %s\n",datfname);
    return -1;
  }

  // Read buffer
  fread(fbuf,sizeof(float),nsamp,file);

  // Close file
  fclose(file);

  // Copy buffer
  for (i=0;i<ndm;i++)
    memcpy(x+i*nsamp,fbuf,sizeof(float)*nsamp);

  // Find number of kernels to convolve
  if (wmax>0.0) {
    for (i=1;i<sizeof(ds)/sizeof(ds[0]);i++) 
      if (ds[i-1]*dt<wmax && ds[i]*dt>wmax)
	break;
    my=i;
  } else {
    for (i=1;i<sizeof(ds)/sizeof(ds[0]);i++) 
      if (ds[i-1]<=dsmax && ds[i]>dsmax)
	break;
    my=i;
  }
   
  // Sizes
  m=8000;
  mdetrend=1000;
  nx=8192;
  mx=nx/2+1;

  // Number of FFTs
  ny=(int) ceil(nsamp/(float) m);

  // Number of detrend lengths
  ndetrend=(int) ceil(nsamp/(float) mdetrend);

  printf("%d samples, %d point fft, %d ffts, %d kernels, %d detrend blocks of %d samples\n",nsamp,nx,ny,my,ndetrend,mdetrend);

  // Allocate device memory for signal
  checkCudaErrors(cudaMalloc((void **) &dxs,sizeof(float)*nsamp*ndm));
  checkCudaErrors(cudaMalloc((void **) &dzs,sizeof(float)*nsamp*ndm));
  checkCudaErrors(cudaMemcpy(dxs,x,sizeof(float)*nsamp*ndm,cudaMemcpyHostToDevice));

  // Detrend timeseries
  blocksize.x=32;blocksize.y=32;blocksize.z=1;
  gridsize.x=ndetrend/blocksize.x+1;gridsize.y=ndm/blocksize.y+1;gridsize.z=1;
  detrend_and_normalize<<<gridsize,blocksize>>>(dxs,dzs,nsamp,mdetrend,ndetrend,ndm);

  // Allocate memory for padded signal
  checkCudaErrors(cudaMalloc((void **) &dx,sizeof(cufftReal)*nx*ny));

  // Padd signal
  blocksize.x=32;blocksize.y=32;blocksize.z=1;
  gridsize.x=nx/blocksize.x+1;gridsize.y=ny/blocksize.y+1;gridsize.z=1;
  padd_data<<<gridsize,blocksize>>>(dxs,dx,nsamp,nx,m,ny);

  // Allocate device memory
  y=(cufftReal *) malloc(sizeof(cufftReal)*nx*my);
  checkCudaErrors(cudaMalloc((void **) &dy,sizeof(cufftReal)*nx*my));
  checkCudaErrors(cudaMalloc((void **) &dz,sizeof(cufftReal)*nx*ny));
  checkCudaErrors(cudaMalloc((void **) &dcx,sizeof(cufftComplex)*mx*ny));
  checkCudaErrors(cudaMalloc((void **) &dcy,sizeof(cufftComplex)*mx*my));
  checkCudaErrors(cudaMalloc((void **) &dcz,sizeof(cufftComplex)*mx*ny));

  // Fill kernel
  for (j=0;j<my;j++) {
    for (i=0;i<nx;i++)
      y[j*nx+i]=0.0;

    if (ds[j]%2!=0) {
      // Odd factors
      for (i=0;i<ds[j]/2+1;i++)
        y[j*nx+i]+=1.0;
      for (i=nx-ds[j]/2;i<nx;i++)
        y[j*nx+i]+=1.0;
    } else {
      // Even factors
      for (i=0;i<ds[j]/2+1;i++)
        y[j*nx+i]+=1.0;
      if (ds[j]>2)
        for (i=nx-ds[j]/2+1;i<nx;i++)
          y[j*nx+i]+=1.0;
    }
    // Divide by sqrt(width)
    for (i=0;i<nx;i++)
      y[j*nx+i]/=sqrt(ds[j]);
  }

  // Copy kernels to device
  checkCudaErrors(cudaMemcpy(dy,y,sizeof(cufftReal)*nx*my,cudaMemcpyHostToDevice));

  // Plan and FFT signal
  idist=nx;  odist=mx;  iembed=nx;  oembed=nx;  istride=1;  ostride=1;
  checkCudaErrors(cufftPlanMany(&ftr2cx,1,&nx,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_R2C,ny));
  checkCudaErrors(cufftExecR2C(ftr2cx,(cufftReal *) dx,(cufftComplex *) dcx));

  // Plan and FFT window
  idist=nx;  odist=mx;  iembed=nx;  oembed=nx;  istride=1;  ostride=1;
  checkCudaErrors(cufftPlanMany(&ftr2cy,1,&nx,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_R2C,my));
  checkCudaErrors(cufftExecR2C(ftr2cy,(cufftReal *) dy,(cufftComplex *) dcy));

  // Free input arrays
  cudaFree(dx);
  cudaFree(dy);

  // Plan convolved signal
  idist=mx;  odist=nx;  iembed=mx;  oembed=mx;  istride=1;  ostride=1;
  checkCudaErrors(cufftPlanMany(&ftc2rz,1,&nx,&iembed,istride,idist,&oembed,ostride,odist,CUFFT_C2R,ny));

  // Allocate mask
  mask=(int *) malloc(sizeof(int)*nsamp);
  checkCudaErrors(cudaMalloc((void **) &dmask,sizeof(int)*nsamp));

  // Allocate width
  w=(int *) malloc(sizeof(int)*nsamp);
  checkCudaErrors(cudaMalloc((void **) &dw,sizeof(int)*nsamp));

  // Set width
  for (i=0;i<nsamp;i++)
    w[i]=1;

  // Copy to device
  checkCudaErrors(cudaMemcpy(dw,w,sizeof(int)*nsamp,cudaMemcpyHostToDevice));

  // Loop over kernels
  for (k=0;k<my;k++) {
    // Complex multiplication
    PointwiseComplexMultiply<<<256,256>>>(dcx,dcy,dcz,mx*ny,mx,k,1.0/(float) nx);
 
    // FFT convolved signal
    checkCudaErrors(cufftExecC2R(ftc2rz,(cufftComplex *) dcz,(cufftReal *) dz));

    // Unpadd convolved signal
    unpadd_data<<<256,256>>>(dzs,dz,nsamp,nx,m,ny);

    // Prune results
    prune<<<256,256>>>(dzs,nsamp,ds[k],dmask,sigma);

    // Store
    store<<<256,256>>>(dxs,dzs,dmask,dw,ds[k],nsamp);
  }

  // Prune final results
  prune_final<<<256,256>>>(dxs,dw,dmask,nsamp,sigma);

  // Store final results
  store_final<<<256,256>>>(dxs,dmask,nsamp,sigma);

  // Copy convolved signal to host
  checkCudaErrors(cudaMemcpy(z,dxs,sizeof(cufftReal)*nsamp,cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(w,dw,sizeof(int)*nsamp,cudaMemcpyDeviceToHost));

  // Open single pulse file
  file=fopen(spfname,"w");
  if (file==NULL) {
    fprintf(stderr,"Error opening %s\n",spfname);
    return -1;
  }
  // Print results
  fprintf(file,"# DM      Sigma      Time (s)     Sample    Downfact   Sampling (s)\n");
  for (i=0,j=0;i<nsamp;i++) {
    if (z[i]>sigma) {
      fprintf(file,"%7.2f %7.2f %13.6f %10d     %3d   %g\n",dm,z[i],i*dt,i,w[i],dt);
      j++;
    }
  }
  fclose(file);
  printf("Found %d candidates\n",j);

  // Destroy plans
  cufftDestroy(ftr2cx);
  cufftDestroy(ftr2cy);
  cufftDestroy(ftc2rz);

  // Free memory
  free(x);
  free(y);
  free(z);
  free(w);
  free(mask);
  cudaFree(dz);
  cudaFree(dcx);
  cudaFree(dcy);
  cudaFree(dcz);
  cudaFree(dxs);
  cudaFree(dw);
  cudaFree(dmask);
  free(datfname);
  free(inffname);
  free(spfname);

  return 0;
}
