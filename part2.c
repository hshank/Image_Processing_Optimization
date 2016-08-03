#define KERNX 3//this is the x-size of the kernel. It will always be odd.
#define KERNY 3//this is the y-size of the kernel. It will always be odd.
#include <emmintrin.h>
#include <omp.h>

int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                   float* kernel)
{
   // the x coordinate of the kernel's center
   int kern_cent_X = (KERNX - 1)/2;
   // the y coordinate of the kernel's center
   int kern_cent_Y = (KERNY - 1)/2;

   int num_threads = 8;

   //padding for input
   int padded_size_X = data_size_X + 2*kern_cent_X;
   while(padded_size_X % 4 != 0){
        padded_size_X++;
   }
int padded_start = kern_cent_X+kern_cent_X*(padded_size_X);	
   int padded_size_Y =  (data_size_Y) + (2*kern_cent_Y);

  //padding for input
  float new_in[padded_size_X * padded_size_Y];
  

  
  #pragma omp parallel
  {



  #pragma omp for
   for(int i = 0; i < padded_size_X * padded_size_Y; i++){

		new_in[i] = 0;
	}

   #pragma omp for
   
   for(int y = 0; y < data_size_Y; y++){
	for(int x = 0; x < data_size_X; x++){
		new_in[(x + kern_cent_X) + (y+kern_cent_Y) * padded_size_X] = in[x + (y)*data_size_X];
	}
   }
	
  // get inverse of kernel
   float inv_kern[KERNX * KERNY];
   for(int x_tmp = 0; x_tmp < KERNX; x_tmp++){
        for(int y_tmp = 0; y_tmp < KERNY; y_tmp++){
                inv_kern[x_tmp + y_tmp*(KERNX)] = kernel[(KERNX - x_tmp - 1) + (KERNY - y_tmp - 1)*KERNX];
        }
   }

   // main convolution loop
        __m128 values, kern, in_vect, kern_mul_in;
        int x, i, j;
        float outTemp;
  	#pragma omp for firstprivate(padded_size_X, data_size_X,data_size_Y) private(x, values, kern, in_vect, kern_mul_in, outTemp) 
        for(int y = 0; y < data_size_Y; y++){ // the y coordinate of the output location we're focusing on
                for(x = 0; x <= data_size_X - 4; x+=4){ // the x coordinate of theoutput location we're focusing on
                        values = _mm_setzero_ps();
                                for(int i = 0; i < KERNX; i++){ // kernel unflipped x coordinate
                        for(int j = 0; j < KERNY; j++){ // kernel unflippaed y coordinate

                                        // only do the operation if not out of bounds
                                                //Note that the kernel is flipped
					kern = _mm_load1_ps(inv_kern + i + j *KERNX);
					in_vect = _mm_loadu_ps(new_in+x+i+ (y+j)*(padded_size_X));
		               		kern_mul_in = _mm_mul_ps(in_vect, kern);
		               		values = _mm_add_ps( values, kern_mul_in );
                                       // values = _mm_add_ps( values, _mm_mul_ps(_mm_load1_ps(inv_kern + i + j *KERNX), _mm_loadu_ps(new_in+x+i+ (y+j)*(padded_size_X))));
                                }

                        }
                        _mm_storeu_ps(out+x+y*data_size_X, values);
                }
                for(; x < data_size_X; x++){
                        outTemp = 0;
                        for(int j = 0; j < KERNY; j++){
                                for(int i= 0; i < KERNX; i++){
                                        outTemp += inv_kern[i + j*KERNX] * new_in[x+i + (y+j)*padded_size_X];
                                }
                        }
                        out[x+y*data_size_X] = outTemp;
                }
	//if (omp_get_thread_num()==15) printf("hello\n");
        }
	//if (omp_get_thread_num()==2) printf("hello\n");
    }
        return 1;
}