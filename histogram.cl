__kernel void histogram(__global unsigned int *image_data,__global unsigned int *ref_histogram_results,unsigned int bound,unsigned int partsize)
{
   int c = get_global_id(0);
   unsigned int i, j;
   unsigned int idx;
   size_t width = get_global_size(0);

   if(c<768) ref_histogram_results[c] = 0;

   for (i=0; i<partsize; i++)
   {
       if ((c+(i*width))<bound)
       {
           for (j = 0; j < 3; j++)
           {
               idx=image_data[(c+(i*width))*3+j];
               atomic_inc(ref_histogram_results+(idx+j*256));
           }
       }
   }
}
