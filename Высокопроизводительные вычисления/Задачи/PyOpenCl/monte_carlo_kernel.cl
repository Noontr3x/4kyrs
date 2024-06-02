__kernel void monte_carlo_pi(__global const int* num_samples, __global float* result, unsigned int seed)
{
    int inside_circle = 0;
    unsigned int local_seed = seed + get_global_id(0);
    for (int i = 0; i < *num_samples; ++i)
    {
        float x = 2.0f * native_random(local_seed) / UINT_MAX - 1.0f;
        float y = 2.0f * native_random(local_seed) / UINT_MAX - 1.0f;
        if (x * x + y * y <= 1.0f)
            inside_circle++;
    }
    result[0] = (float)inside_circle / *num_samples * 4.0f;
}
