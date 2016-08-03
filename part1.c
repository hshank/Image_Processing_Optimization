#define KERNX 3//this is the x-size of the kernel. It will always be odd.
#define KERNY 3//this is the y-size of the kernel. It will always be odd.
#include <emmintrin.h>#include <emmintrin.h>

int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                   float* kernel)
{
   // the x coordinate of the kernel's center
   int kern_cent_X = (KERNX - 1)/2;
   // the y coordinate of the kernel's center
   int kern_cent_Y = (KERNY - 1)/2;

   //padding for input
   int padded_size_X = data_size_X + 2*kern_cent_X;
   while(padded_size_X % 4 != 0){
        padded_size_X++;
   }
   int padded_size_Y =  (data_size_Y) + (2*kern_cent_Y);

  //padding for input
  float new_in[padded_size_X * padded_size_Y];
   for(int x_var = 0; x_var< padded_size_X; x_var++){
        for(int y_var = 0; y_var< padded_size_Y; y_var++){
		if (x_var < kern_cent_X || y_var < kern_cent_Y || x_var >= data_size_X + kern_cent_X  || y_var >= data_size_Y + kern_cent_Y){
			new_in[x_var+ y_var*padded_size_X] = 0;
		} else {
			new_in[x_var+ y_var*padded_size_X] = in[x_var - kern_cent_X + (y_var - kern_cent_Y)*data_size_X];
		}
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
        __m128 values;
        int x, i, j;
        float outTemp;

        for(int y = 0; y < data_size_Y; y++){ // the y coordinate of the output location we're focusing on
                for(x = 0; x <= data_size_X - 4; x+=4){ // the x coordinate of theoutput location we're focusing on
                        outTemp = 0;
                        values = _mm_setzero_ps();
                        for(j = 0; j < KERNY; j++){ // kernel unflippaed y coordinate
                                for(i = 0; i < KERNX; i++){ // kernel unflipped x coordinate
                                        // only do the operation if not out of bounds
                                                //Note that the kernel is flipped
                                        values = _mm_add_ps( values, _mm_mul_ps(_mm_load1_ps(inv_kern + i + j *KERNX), _mm_loadu_ps(new_in+x+i+ (y+j)*(padded_size_X))));
                                }

                        }
                        _mm_storeu_ps(out+x+y*data_size_X, values);
                }
                for(; x < data_size_X; x++){
                        outTemp = 0;
                        for(j = 0; j < KERNY; j++){
                                for(i= 0; i < KERNX; i++){
                                        outTemp += inv_kern[i + j*KERNX] * new_in[x+i + (y+j)*padded_size_X];
                                }
                        }
                        out[x+y*data_size_X] = outTemp;
                }
        }
        return 1;
}