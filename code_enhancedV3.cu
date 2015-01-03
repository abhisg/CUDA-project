#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <stack>
#include <map>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#define radius 2
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

using namespace std;

struct corners
{
	int x;
	int xbar;
	int y;
};

__global__ void calculateDerivatives(unsigned char *image,float *Ix,float *Iy,int m,int n)
{
	int j = blockIdx.y*blockDim.y+threadIdx.y;
	int i = blockIdx.x*blockDim.x+threadIdx.x;

	if(i<m && j<n)
	{
		int cl = j-radius;
		if (cl<0)
			cl = 0;
		int cr = j+radius;
		if(cr>=n)
			cr = n-1;
		int ru = i-radius;
		if(ru<0)
			ru = 0;
		int rl = i+radius;
		if(rl>=m)	
			rl = m-1;
		int ix = 0,iy = 0;
		for(int i1=ru;i1<=rl;i1++)
		{
			for(int j1=cl;j1<=cr;j1++)
			{
				if(i1<i)
					iy = iy - image[i1*n+j1];
				else if(i1>i)
					iy = iy + image[i1*n+j1];
				if(j1<j)
					ix = ix - image[i1*n+j1];
				else if(j1>j)
					ix = ix + image[i1*n+j1];
			}
		}
		Ix[i*n+j] = ix;
		Iy[i*n+j] = iy;
	}
}

__global__ void computeStrength(unsigned char *image,float *kernel,float *strength,float *Ix,float *Iy,int m,int n)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;

	if(i<m && j<n)
	{
		float ix_2=0,iy_2=0,ixy=0;
		int cl = j-radius;
		if (cl<0)
			cl = 0;
		int cr = j+radius;
		if(cr>=n)
			cr = n-1;
		int ru = i-radius;
		if(ru<0)
			ru = 0;
		int rl = i+radius;
		if(rl>=m)	
			rl = m-1;
		for(int i1=ru;i1<=rl;i1++)
		{
			for(int j1=cl;j1<=cr;j1++)
			{
				ix_2 = ix_2 + kernel[(i1-ru)*(2*radius+1)+(j1-cl)]*Ix[i1*n+j1]*Ix[i1*n+j1];
				iy_2 = iy_2 + kernel[(i1-ru)*(2*radius+1)+(j1-cl)]*Iy[i1*n+j1]*Iy[i1*n+j1];	
				ixy = ixy + kernel[(i1-ru)*(2*radius+1)+(j1-cl)]*Ix[i1*n+j1]*Iy[i1*n+j1];
			}
		}
		strength[i*n+j] = (ix_2*iy_2-pow(ixy,2))/(ix_2+iy_2+0.0001);
	}
}

__global__ void harrisCorner(float *strength,uchar *corner,int m,int n,float thres)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j = blockIdx.y*blockDim.y+threadIdx.y;
	
	if(i<m && j<n)
	{
		int cl = j-radius;
		if (cl<0)
			cl = 0;
		int cr = j+radius;
		if(cr>=n)
			cr = n-1;
		int ru = i-radius;
		if(ru<0)
			ru = 0;
		int rl = i+radius;
		if(rl>=m)	
			rl = m-1;
		int maxX = cl,maxY = ru;
		float maxVal = 0;
		for(int i1=ru;i1<=rl;i1++)
		{
			for(int j1=cl;j1<=cr;j1++)
			{
				if(strength[i1*n+j1]>maxVal || (strength[i1*n+j1]==maxVal && i1==i && j1==j))
				{
					maxVal = strength[i1*n+j1];
					maxX = j1;
					maxY = i1;
				}
			}
		}
		if(maxX==j && maxY==i && maxVal>thres)
			corner[i*n+j] = '1';
		else
			corner[i*n+j] = '0'; 
	}
}
		
__device__  float PYTHAG(float a, float b)
{
	float at = fabs(a), bt = fabs(b), ct, result;

	if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
	else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
	else result = 0.0;
	return(result);
}


__device__ int dsvd(float a[][9], int m, int n, float *w, float v[][9])
{
	int flag, i, its, j, jj, k, l, nm;
	float c, f, h, s, x, y, z;
	float anorm = 0.0, g = 0.0, scale = 0.0;
	float rv1[9];
	/* Householder reduction to bidiagonal form */
	for (i = 0; i < n; i++) 
	{
		/* left-hand reduction */
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m) 
		{
			for (k = i; k < m; k++) 
			{
				if(a[k*m+i]>=0)
					scale += a[k][i];
				else
					scale -= a[k][i];
			}
			if (scale) 
			{
				for (k = i; k < m; k++) 
				{
					a[k][i] = a[k][i]/scale;
					s += (a[k][i] * a[k][i]);
				}
				f = a[i][i];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][i] = (float)(f - g);
				if (i != n - 1) 
				{
					for (j = l; j < n; j++) 
					{
						for (s = 0.0, k = i; k < m; k++) 
							s += ((double)a[k][i] * (double)a[k][j]);
						f = s / h;
						for (k = i; k < m; k++) 
							a[k][j] += (float)(f * (double)a[k][i]);
					}
				}
				for (k = i; k < m; k++) 
					a[k][i] = (float)((double)a[k][i]*scale);
			}
		}
		w[i] = (float)(scale * g);

		/* right-hand reduction */
		g = s = scale = 0.0;
		if (i < m && i != n - 1) 
		{
			for (k = l; k < n; k++) 
				scale += fabs((double)a[i][k]);
			if (scale) 
			{
				for (k = l; k < n; k++) 
				{
					a[i][k] = (float)((double)a[i][k]/scale);
					s += ((double)a[i][k] * (double)a[i][k]);
				}
				f = (double)a[i][l];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][l] = (float)(f - g);
				for (k = l; k < n; k++) 
					rv1[k] = (double)a[i][k] / h;
				if (i != m - 1) 
				{
					for (j = l; j < m; j++) 
					{
						for (s = 0.0, k = l; k < n; k++) 
							s += ((double)a[j][k] * (double)a[i][k]);
						for (k = l; k < n; k++) 
							a[j][k] += (float)(s * rv1[k]);
					}
				}
				for (k = l; k < n; k++) 
					a[i][k] = (float)((double)a[i][k]*scale);
			}
		}
		anorm = MAX(anorm, (fabs((double)w[i]) + fabs(rv1[i])));
	}

	/* accumulate the right-hand transformation */
	for (i = n - 1; i >= 0; i--) 
	{
		if (i < n - 1) 
		{
			if (g) 
			{
				for (j = l; j < n; j++)
					v[j][i] = (float)(((double)a[i][j] / (double)a[i][l]) / g);
				/* double division to avoid underflow */
				for (j = l; j < n; j++) 
				{
					for (s = 0.0, k = l; k < n; k++) 
						s += ((double)a[i][k] * (double)v[k][j]);
					for (k = l; k < n; k++) 
						v[k][j] += (float)(s * (double)v[k][i]);
				}
			}
			for (j = l; j < n; j++) 
				v[i][j] = v[j][i] = 0.0;
		}
		v[i][i] = 1.0;
		g = rv1[i];
		l = i;
	}

	/* accumulate the left-hand transformation */
	for (i = n - 1; i >= 0; i--) 
	{
		l = i + 1;
		g = (double)w[i];
		if (i < n - 1) 
			for (j = l; j < n; j++) 
				a[i][j] = 0.0;
		if (g) 
		{
			g = 1.0 / g;
			if (i != n - 1) 
			{
				for (j = l; j < n; j++) 
				{
					for (s = 0.0, k = l; k < m; k++) 
						s += ((double)a[k][i] * (double)a[k][j]);
					f = (s / (double)a[i][i]) * g;
					for (k = i; k < m; k++) 
						a[k][j] += (float)(f * (double)a[k][i]);
				}
			}
			for (j = i; j < m; j++) 
				a[j][i] = (float)((double)a[j][i]*g);
		}
		else 
		{
			for (j = i; j < m; j++) 
				a[j][i] = 0.0;
		}
		++a[i][i];
	}

	/* diagonalize the bidiagonal form */
	for (k = n - 1; k >= 0; k--) 
	{                             /* loop over singular values */
		for (its = 0; its < 100; its++) 
		{                         /* loop over allowed iterations */
			flag = 1;
			for (l = k; l >= 0; l--) 
			{                     /* test for splitting */
				nm = l - 1;
				if (fabs(rv1[l]) + anorm == anorm) 
				{
					flag = 0;
					break;
				}
				if (fabs((double)w[nm]) + anorm == anorm) 
					break;
			}
			if (flag) 
			{
				c = 0.0;
				s = 1.0;
				for (i = l; i <= k; i++) 
				{
					f = s * rv1[i];
					if (fabs(f) + anorm != anorm) 
					{
						g = (double)w[i];
						h = PYTHAG(f, g);
						w[i] = (float)h; 
						h = 1.0 / h;
						c = g * h;
						s = (- f * h);
						for (j = 0; j < m; j++) 
						{
							y = (double)a[j][nm];
							z = (double)a[j][i];
							a[j][nm] = (float)(y * c + z * s);
							a[j][i] = (float)(z * c - y * s);
						}
					}
				}
			}
			z = (double)w[k];
			if (l == k) 
			{                  /* convergence */
				if (z < 0.0) 
				{              /* make singular value nonnegative */
					w[k] = (float)(-z);
					for (j = 0; j < n; j++) 
						v[j][k] = (-v[j][k]);
				}
				break;
			}
			if (its >= 100) {
				return 0;
			}

			/* shift from bottom 2 x 2 minor */
			x = (double)w[l];
			nm = k - 1;
			y = (double)w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = PYTHAG(f, 1.0);
			f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

			/* next QR transformation */
			c = s = 1.0;
			for (j = l; j <= nm; j++) 
			{
				i = j + 1;
				g = rv1[i];
				y = (double)w[i];
				h = s * g;
				g = c * g;
				z = PYTHAG(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y = y * c;
				for (jj = 0; jj < n; jj++) 
				{
					x = (double)v[jj][j];
					z = (double)v[jj][i];
					v[jj][j] = (float)(x * c + z * s);
					v[jj][i] = (float)(z * c - x * s);
				}
				z = PYTHAG(f, h);
				w[j] = (float)z;
				if (z) 
				{
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = (c * g) + (s * y);
				x = (c * y) - (s * g);
				for (jj = 0; jj < m; jj++) 
				{
					y = (double)a[jj][j];
					z = (double)a[jj][i];
					a[jj][j] = (float)(y * c + z * s);
					a[jj][i] = (float)(z * c - y * s);
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = (float)x;
		}
	}
	return(1);
}

__device__ void computeNCC(struct corners points[][5],uchar *leftimage,uchar *rightimage,float *ncc,int m,int n,uchar bookmark[][5],int neighbourhood)
{
	float sum = 0;
	float sq1 = 0;
	float sq2 = 0;
	for(int i=0;i<5;i++)
	{
		if(bookmark[neighbourhood][i]=='1')
		{
			struct corners p =  points[neighbourhood][i];
			sum += leftimage[p.x*n+p.y]*rightimage[p.xbar*n+p.y];
			sq1 += pow((float)leftimage[p.x*n+p.y],2);
			sq2 += pow((float)rightimage[p.xbar*n+p.y],2);
		}
	}
	if(sq1!=0 && sq2!=0)
		ncc[neighbourhood] = sum/sqrt(sq1*sq2);
	else
		ncc[neighbourhood] = 0;
}

__global__ void DFS(thrust::device_vector<struct corners> DFSpoints,int *start,int *end,uchar *left_d,uchar *right_d,int m,int n)
{
	int node =  blockIdx.x*blockDim.x+threadIdx.x;
	if(*start+node<*end)
	{
		int modif[16][3] = {{-1,-1,0},{-2,-1,0},{-1,-2,0},{1,1,0},{2,1,0},{1,2,0},{0,0,-1},{1,0,-1},{-1,0,-1},{0,1,-1},{0,1,-1},{0,0,1},{1,0,1},{-1,0,1},{0,1,1},{0,-1,1}}; 
		uchar bookmark[4][5]={{'0','0','0','0','0'},{'0','0','0','0','0'},{'0','0','0','0','0'},{'0','0','0','0','0'}};
		struct corners point=DFSpoints[node];
		float score[4];
		struct corners expansion[4][5];
		int counter=0;
		for(int i=0;i<4;i++)
		{
			int limit = i<2?3:5;
			for(int j=0;j<limit;j++)
			{
				struct corners newpt = point;
				if(newpt.x+modif[counter][0]< n && newpt.xbar+modif[counter][1] < n && newpt.y+modif[counter][2]<m)
				{
					newpt.x += modif[counter][0];
					newpt.xbar += modif[counter][1];
					newpt.y += modif[counter][2];
					expansion[i][j] = newpt;
					bookmark[i][j]='1';
				}
				counter++;
			}
			computeNCC(expansion,left_d,right_d,score,m,n,bookmark,i);
		}
		float bestScore = score[0];
		int bestIndex = 0;
		for(int i=1;i<4;i++)
		{
			if(score[i] > bestScore)
			{
				bestScore = score[i];
				bestIndex = i;
			}
		}
		float thres = 0.99999; 
		if(bestScore > thres)
		{
			for(int i=0;i<5;i++)
			{
				if(bookmark[bestIndex][i]=='1')
				{
					struct corners pt = expansion[bestIndex][i];
					int j;
					for(j=*start;j<*end;j++)
					{
						thrust::device_reference<struct corners> temp = DFSpoints[j];
						if(pt.x==temp->x && pt.y==temp->y && pt.xbar==temp->xbar)
							break;
					}
					if(j==*end)
						DFSpoints.push_back(pt);
				}
			}
		}
	}
}

__global__ void homographyFilter(struct corners *filteredCorners_d,uchar *agreeingCorners_d,int *scoreOfAgreement_d,uchar *left_d,uchar *right_d,int m,int n,int N,int valid)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<N)
	{
		float left[3][4],right[3][4];
		float meanX=0,meanY=0,meanXbar=0;
		for(int j=0;j<4;j++)
		{
			left[0][j] = filteredCorners_d[4*i+j].x;
			left[1][j] = filteredCorners_d[4*i+j].y;
			left[2][j] = 1;
			right[0][j] = filteredCorners_d[4*i+j].xbar;
			right[1][j] = filteredCorners_d[4*i+j].y;
			right[2][j] = 1;
			meanX = meanX + filteredCorners_d[4*i+j].x; 
			meanXbar = meanXbar + filteredCorners_d[4*i+j].xbar;
			meanY = meanY + filteredCorners_d[4*i+j].y; 
		}	
		meanX = meanX*1.0/4;
		meanY = meanY*1.0/4;
		meanXbar = meanXbar*1.0/4;
		float scaleLeft=0.0,scaleRight=0.0;
		for(int j=0;j<4;j++)
		{
			scaleLeft += sqrt(pow(left[0][j],2)+pow(left[1][j],2));		
			scaleRight += sqrt(pow(right[0][j],2)+pow(right[1][j],2));
		}
		scaleLeft = 4*sqrt(2.0)/scaleLeft;
		scaleRight = 4*sqrt(2.0)/scaleRight;
		float T_left[3][3]={{scaleLeft,0,-scaleLeft*meanX},{0,scaleLeft,-scaleLeft*meanY},{0,0,1}}; 	
		float T_right[3][3]={{scaleRight,0,-scaleRight*meanXbar},{0,scaleRight,-scaleRight*meanY},{0,0,1}}; 	
		for(int j=0;j<4;j++)
		{
			left[0][j] = scaleLeft*left[0][j]-scaleLeft*meanX;
			left[1][j] = scaleLeft*left[1][j]-scaleLeft*meanY;
			right[0][j] = scaleRight*right[0][j]-scaleRight*meanXbar;
			right[1][j] = scaleRight*right[1][j]-scaleRight*meanY;
		}
		float A[12][9],U[9],V[9][9];
		for(int j=0;j<4;j++)
		{
			for(int k=0;k<3;k++)
				A[3*j][k]=0;
			for(int k=3;k<6;k++)
				A[3*j][k] = -right[k-3][j];
			for(int k=6;k<9;k++)
				A[3*j][k] = left[1][j]*right[k-6][j];
			for(int k=0;k<3;k++)
				A[3*j+1][k] = right[k][j];
			for(int k=3;k<6;k++)
				A[3*j+1][k] = 0;
			for(int k=6;k<9;k++)
				A[3*j+1][j] = -left[0][j]*right[k-6][j];
			for(int k=0;k<3;k++)
				A[3*j+2][k] = -left[1][j]*right[k][j];
			for(int k=3;k<6;k++)
				A[3*j+2][k] = left[0][j]*right[k-3][j];
			for(int k=6;k<9;k++)
				A[3*j+2][j] = 0;
		}
		dsvd(A,12,9,U,V);	
		float t1[3][3]={{1/scaleLeft,0,meanX},{0,1/scaleLeft,meanY},{0,0,1}};
		float oldH[3][3] ={{V[0][8],V[1][8],V[2][8]},{V[3][8],V[4][8],V[5][8]},{V[6][8],V[7][8],V[8][8]}};
		float temp1[3][3];
		for(int j=0;j<3;j++)
		{
			for(int k=0;k<3;k++)
			{
				temp1[j][k] = 0;
				for(int l=0;l<3;l++)
					temp1[j][k] += t1[j][l]*oldH[l][k];
			}
		}
		float H[3][3];
		for(int j=0;j<3;j++)
		{
			for(int k=0;k<3;k++)
			{
				H[j][k] = 0;
				for(int l=0;l<3;l++)
					H[j][k] += temp1[j][l]*T_right[l][k];
			}
		}
		scoreOfAgreement_d[i] = 0;	
		for(int j=0;j<valid;j++)
		{
			float dist = 0;
			float newX = H[0][0]*filteredCorners_d[j].xbar+H[0][1]*filteredCorners_d[j].y+H[0][2];
			float newY = H[1][0]*filteredCorners_d[j].xbar+H[1][1]*filteredCorners_d[j].y+H[1][2];
			float newZ = H[2][0]*filteredCorners_d[j].xbar+H[2][1]*filteredCorners_d[j].y+H[2][2];
			dist = sqrt(pow(newX-filteredCorners_d[j].x,2)+pow(newY-filteredCorners_d[j].y,2));
			if (dist<100)
			{
				scoreOfAgreement_d[i]++;
				agreeingCorners_d[i*valid+j] = '1';
			}
		}
	}
}


__global__ void matchCorners(struct corners *Corners1,struct corners *Corners2,int N1,int N2,int m,int n,uchar *image1,uchar *image2)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;

	if(i < N1)
	{
		int x = Corners1[i].x;
		int y = Corners1[i].y;
		float bestMatch = -10000000000; 
		for(int j=0;j<N2;j++)
		{
			if(Corners2[j].y == y)
			{
				int xbar = Corners2[j].x;
				int count = 0;
				float avgI = 0;
				float avgIbar = 0;
				float varI = 0;
				float varIbar = 0;
				for(int i1 = -radius;i1<=radius;i1++)
				{
					for(int j1=-radius;j1<=radius;j1++)
					{
						if(x+j1>=0 && x+j1<n && xbar+j1>=0 && xbar+j1<n && y+i1>=0 && y+i1<m)
						{
							count++;
							avgI = avgI +  image1[(y+i1)*n+x+j1];
							avgIbar = avgIbar + image2[(y+i1)*n+xbar+j1];
						}
					}
				}
				if(count!=0)
				{
					avgI = avgI*1.0/count;
					avgIbar = avgIbar*1.0/count;
					for(int i1 = -radius;i1<=radius;i1++)
					{
						for(int j1=-radius;j1<=radius;j1++)
						{
							if(x+j1>=0 && x+j1<n && xbar+j1>=0 && xbar+j1<n && y+i1>=0 && y+i1<m)
							{
								varI = varI +  pow(image1[(y+i1)*n+x+j1]-avgI,2);
								varIbar = varIbar + pow(image2[(y+i1)*n+xbar+j1]-avgIbar,2);
							}
						}
					}
					varI = varI*1.0/count;
					varIbar = varIbar*1.0/count;
					float score = 0;
					for(int i1 = -radius;i1<=radius;i1++)
					{
						for(int j1=-radius;j1<=radius;j1++)
						{
							if(x+j1>=0 && x+j1<n && xbar+j1>=0 && xbar+j1<n && y+i1>=0 && y+i1<m)
								score = score +  (image1[(y+i1)*n+x+j1] - avgI)*(image2[(y+i1)*n+xbar+j1]-avgIbar);				
						}
					}
					if(varI!=0 && varIbar!=0)
					{
						score = score/sqrt(varI*varIbar);
						if(score>bestMatch)
						{
							bestMatch = score;
							Corners1[i].xbar = Corners2[j].x;
						}
					}
				}
			}
		}
	}
}

__global__ void filterCorners(struct corners *Corners1,struct corners *Corners2,int N_left,int N_right,uchar *bitmap)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<N_left)
	{
		int x = Corners1[i].x;
		int y = Corners1[i].y;
		int xbar = Corners1[i].xbar;
		int j;
		for(j=0;j<N_right;j++)
		{
			if(Corners2[j].xbar == x && Corners2[j].x == xbar && Corners2[j].y == y)
				break;
		}
		if(j!=N_right)
			bitmap[i] = '1';
		else
			bitmap[i] = '0';
	}
}

//do most of the work in another function.call a set of images in the main function and analyse the results in the main function itself

int main()
{
	cv::Mat leftImg = cv::imread("epig.tif");
	cv::Mat rightImg = cv::imread("epid.tif");
	cv::Mat kernel1 = cv::getGaussianKernel(2*radius+1,5.5);
	cv::Mat kernel2 = cv::getGaussianKernel(2*radius+1,5.5);
	cv::Mat kernel = kernel1*kernel2.t();
	float *kernel_h = (float *)malloc((2*radius+1)*(2*radius+1)*sizeof(float));
	for(int i=0;i<2*radius+1;i++)
	{
		for(int j=0;j<2*radius+1;j++)
			kernel_h[i*(2*radius+1)+j] = (float)kernel.at<double>(i,j);
	}

	int m_left = leftImg.rows;
	int n_left = leftImg.cols;
	int m_right = rightImg.rows;
	int n_right = rightImg.cols;

	if(leftImg.channels()>=3)
		cv::cvtColor(leftImg,leftImg,CV_BGR2GRAY);
	if(rightImg.channels()>=3)
		cv::cvtColor(rightImg,rightImg,CV_BGR2GRAY);

	unsigned char *left_d,*left_h,*right_d,*right_h,*leftcorners_d,*leftcorners_h,*rightcorners_d,*rightcorners_h;
	float *Ix_d,*Iy_d,*leftstrength_d,*rightstrength_d;
	float *kernel_d;

	size_t sizeLeftImg = m_left*n_left*sizeof(char);
	size_t sizeRightImg = m_right*n_right*sizeof(char);
	size_t sizeLeftDer = m_left*n_left*sizeof(float);
	size_t sizeRightDer = m_right*n_right*sizeof(float);

	left_h = (unsigned char *)malloc(sizeLeftImg);
	for(int i=0;i<m_left;i++)
	{
		for(int j=0;j<n_left;j++)
			left_h[i*n_left+j] = leftImg.at<unsigned char>(i,j);
	}
	printf("%d\n",cudaMalloc((void **)&left_d,sizeLeftImg));
	printf("%d\n",cudaMemcpy(left_d,left_h,sizeLeftImg,cudaMemcpyHostToDevice));

	printf("%d\n",cudaMalloc((void **)&leftstrength_d,sizeLeftImg*sizeof(float)));  	
	printf("%d\n",cudaMalloc((void **)&leftcorners_d,sizeLeftImg));  	
	printf("%d\n",cudaMalloc((void **)&Ix_d,sizeLeftDer));  	
	printf("%d\n",cudaMalloc((void **)&Iy_d,sizeLeftDer));

	printf("%d\n",cudaMalloc((void **)&kernel_d,(2*radius+1)*(2*radius+1)*sizeof(float)));  	
	printf("%d\n",cudaMemcpy(kernel_d,kernel_h,(2*radius+1)*(2*radius+1)*sizeof(float),cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(16,16);
	dim3 numBlocks((unsigned int)ceil(m_left*1.0/threadsPerBlock.x),(unsigned int)ceil(n_left*1.0/threadsPerBlock.y));

	calculateDerivatives<<<numBlocks,threadsPerBlock>>>(left_d,Ix_d,Iy_d,m_left,n_left);
	printf("%d\n",cudaDeviceSynchronize());
	computeStrength<<<numBlocks,threadsPerBlock>>>(left_d,kernel_d,leftstrength_d,Ix_d,Iy_d,m_left,n_left);
	printf("%d\n",cudaDeviceSynchronize());
	harrisCorner<<<numBlocks,threadsPerBlock>>>(leftstrength_d,leftcorners_d,m_left,n_left,15000);	
	printf("%d\n",cudaDeviceSynchronize());

	leftcorners_h = (uchar *)malloc(sizeLeftImg);
	printf("%d\n",cudaMemcpy(leftcorners_h,leftcorners_d,sizeLeftImg,cudaMemcpyDeviceToHost));
	int lc=0,rc=0;
	for(int i=0;i<m_left;i++)
	{
		for(int j=0;j<n_left;j++)
		{
			if(leftcorners_h[i*n_left+j]!='0')
				lc++;
		}
	}

	printf("%d\n",cudaFree(Ix_d));
	printf("%d\n",cudaFree(Iy_d));
	right_h = (unsigned char *)malloc(sizeRightImg);
	for(int i=0;i<m_right;i++)
	{
		for(int j=0;j<n_right;j++)
			right_h[i*n_right+j] = rightImg.at<unsigned char>(i,j);
	}

	printf("%d\n",cudaMalloc((void **)&right_d,sizeRightImg));
	printf("%d\n",cudaMemcpy(right_d,right_h,sizeRightImg,cudaMemcpyHostToDevice));

	printf("%d\n",cudaMalloc((void **)&rightstrength_d,sizeRightImg*sizeof(float)));  	
	printf("%d\n",cudaMalloc((void **)&rightcorners_d,sizeRightImg));  	
	printf("%d\n",cudaMalloc((void **)&Ix_d,sizeRightDer));  	
	printf("%d\n",cudaMalloc((void **)&Iy_d,sizeRightDer));

	calculateDerivatives<<<numBlocks,threadsPerBlock>>>(right_d,Ix_d,Iy_d,m_right,n_right);
	printf("%d\n",cudaDeviceSynchronize());
	computeStrength<<<numBlocks,threadsPerBlock>>>(right_d,kernel_d,rightstrength_d,Ix_d,Iy_d,m_right,n_right);
	printf("%d\n",cudaDeviceSynchronize());
	harrisCorner<<<numBlocks,threadsPerBlock>>>(rightstrength_d,rightcorners_d,m_right,n_right,15000);	
	printf("%d\n",cudaDeviceSynchronize());

	rightcorners_h = (uchar *)malloc(sizeRightImg);
	printf("%d\n",cudaMemcpy(rightcorners_h,rightcorners_d,sizeRightImg,cudaMemcpyDeviceToHost));
	for(int i=0;i<m_right;i++)
	{
		for(int j=0;j<n_right;j++)
		{
			if(rightcorners_h[i*n_right+j]!='0')
				rc++;
		}
	}
	cudaFree(Ix_d);
	cudaFree(Iy_d);
	cudaFree(kernel_d);

	//Match the corners
	struct corners *lft_h,*lft_d,*rgh_h,*rgh_d;
	printf("%d\n",cudaMalloc((void **)&lft_d,lc*sizeof(struct corners)));
	printf("%d\n",cudaMalloc((void **)&rgh_d,rc*sizeof(struct corners)));

	lft_h = (struct corners *)malloc(lc*sizeof(struct corners));
	rgh_h = (struct corners *)malloc(rc*sizeof(struct corners));

	int count = 0;
	for(int i=0;i<m_left;i++)
	{
		for(int j=0;j<n_left;j++)
		{
			if(leftcorners_h[i*n_left+j]!='0')
			{
				lft_h[count].y = i;
				lft_h[count].x = j;
				count++;
			}
		}
	}
	count = 0;			  
	for(int i=0;i<m_right;i++)
	{
		for(int j=0;j<n_right;j++)
		{
			if(rightcorners_h[i*n_right+j]!='0')
			{
				rgh_h[count].y = i;
				rgh_h[count].x = j;
				count++;
			}
		}
	}
	printf("%d\n",cudaMemcpy(lft_d,lft_h,lc*sizeof(struct corners),cudaMemcpyHostToDevice));
	printf("%d\n",cudaMemcpy(rgh_d,rgh_h,rc*sizeof(struct corners),cudaMemcpyHostToDevice));

	dim3 newthreadsPerBlock(256,1,1);
	dim3 newnumberOfBlocks((unsigned int)ceil(lc*1.0/newthreadsPerBlock.x),1,1);
	matchCorners<<<newnumberOfBlocks,newthreadsPerBlock>>>(lft_d,rgh_d,lc,rc,m_left,n_left,left_d,right_d);
	printf("%d\n",cudaDeviceSynchronize());
	cudaMemcpy(lft_h,lft_d,lc*sizeof(struct corners),cudaMemcpyDeviceToHost);
	matchCorners<<<newnumberOfBlocks,newthreadsPerBlock>>>(rgh_d,lft_d,rc,lc,m_right,n_right,right_d,left_d);
	printf("%d\n",cudaDeviceSynchronize());
	cudaMemcpy(lft_h,lft_d,lc*sizeof(struct corners),cudaMemcpyDeviceToHost);

	uchar *bitmap_d,*bitmap_h;
	cudaMalloc((void **)&bitmap_d,lc*sizeof(struct corners));  
	filterCorners<<<newnumberOfBlocks,newthreadsPerBlock>>>(lft_d,rgh_d,lc,rc,bitmap_d);
	printf("%d\n",cudaDeviceSynchronize());
	bitmap_h = (uchar *)malloc(lc*sizeof(uchar));
	cudaMemcpy(bitmap_h,bitmap_d,lc*sizeof(uchar),cudaMemcpyDeviceToHost);
	printf("%d\n",cudaDeviceSynchronize());

	int validCorners = 0;
	for(int i=0;i<lc;i++)
	{
		if(bitmap_h[i]=='1')
			validCorners++;
	}
	cvtColor(leftImg,leftImg,CV_GRAY2BGR);
	cvtColor(rightImg,rightImg,CV_GRAY2BGR);
	struct corners *filteredCorners_h,*filteredCorners_d;
	filteredCorners_h = (struct corners *)malloc(validCorners * sizeof(struct corners));
	int counter = 0;
	cv::Point p; 	
	for(int i=0;i<lc;i++)
	{
		if(bitmap_h[i]=='1')
		{
	
			/*p.x = lft_h[i].x;
			p.y = lft_h[i].y;
			if(i%3==0)
				circle(leftImg,p,5,cv::Scalar(i,0,0),2);
			else if(i%3==1)
				circle(leftImg,p,5,cv::Scalar(0,i,0),2);
			else
				circle(leftImg,p,5,cv::Scalar(0,0,i),2);
			p.x = lft_h[i].xbar;
			if(i%3==0)
				circle(rightImg,p,5,cv::Scalar(i,0,0),2);
			else if(i%3==1)
				circle(rightImg,p,5,cv::Scalar(0,i,0),2);
			else
				circle(rightImg,p,5,cv::Scalar(0,0,i),2);*/
			filteredCorners_h[counter++] = lft_h[i];
		}	
	}
	//deal with the homographies
	int numberOfHomographies = (validCorners/4);
	dim3 threadsForRansac(128,1,1);
	dim3 blocksForRansac((unsigned int)ceil(numberOfHomographies*1.0/threadsForRansac.x),1,1);
	printf("%d\n",cudaMalloc((void **)&filteredCorners_d,validCorners*sizeof(struct corners)));
	printf("%d\n",cudaMemcpy(filteredCorners_d,filteredCorners_h,validCorners*sizeof(struct corners),cudaMemcpyHostToDevice));

	uchar *agreeingCorners_h,*agreeingCorners_d;
	agreeingCorners_h = (uchar *)malloc(numberOfHomographies*validCorners*sizeof(uchar));
	printf("%d\n",cudaMalloc((void **)&agreeingCorners_d,numberOfHomographies*validCorners*sizeof(uchar)));
	int *scoreOfAgreement_h,*scoreOfAgreement_d;
	scoreOfAgreement_h = (int *)malloc(numberOfHomographies*sizeof(int));
	printf("%d\n",cudaMalloc((void **)&scoreOfAgreement_d,numberOfHomographies*sizeof(int)));

	homographyFilter<<<blocksForRansac,threadsForRansac>>>(filteredCorners_d,agreeingCorners_d,scoreOfAgreement_d,left_d,right_d,m_left,n_left,numberOfHomographies,validCorners);
	printf("%d\n",cudaDeviceSynchronize());
	printf("%d\n",cudaMemcpy(scoreOfAgreement_h,scoreOfAgreement_d,numberOfHomographies*sizeof(int),cudaMemcpyDeviceToHost));
	printf("%d\n",cudaMemcpy(agreeingCorners_h,agreeingCorners_d,numberOfHomographies*validCorners*sizeof(uchar),cudaMemcpyDeviceToHost));

	int bestIndex = 0,bestScore = scoreOfAgreement_h[0];
	for(int i = 1;i<numberOfHomographies;i++)
	{
		printf("%d %d\n",validCorners,scoreOfAgreement_h[i]);
		if(scoreOfAgreement_h[i] > bestScore)
		{
			bestIndex = i;
			bestScore = scoreOfAgreement_h[i];
		}
	}
	printf("%d\n",bestIndex);
	thrust::host_vector<struct corners> DFSpoints_h,DFSpoints_d;
	for(int i=0;i<validCorners;i++)
	{
		if(agreeingCorners_h[bestIndex*validCorners+i] == '1')
		{
			struct corners pt = filteredCorners_h[i];
			DFSpoints_h.push_back(pt);
		}
	}
	int start=0,end=DFSpoints_h.size();
	DFSpoints_d = DFSpoints_h;
	while(true)
	{
		dim3 threadsForDFS(32,1,1);
		dim3 blocksForDFS((unsigned int)ceil((end-start)*1.0/threadsForDFS.x),1,1);
		DFS<<<blocksForDFS,threadsForDFS>>>(DFSpoints_d,&start,&end,left_d,right_d,m_left,n_left);
		if(end == DFSpoints_d.size())
			break;
		start = end;
		end=DFSpoints_d.size();
	}
	DFSpoints_h = DFSpoints_d;	
	printf("%d\n",DFSpoints_h.size());
	return 0;	
}
