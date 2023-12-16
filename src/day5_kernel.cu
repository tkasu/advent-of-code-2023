struct LocMapComponent {
    unsigned int from;
    unsigned int to;
    int offset;
};

struct Garden {
    unsigned int *seed_input;
    unsigned int seed_input_size;
    LocMapComponent *seed_to_soil;
    unsigned int seed_to_soil_size;
    LocMapComponent *soil_to_fertilizer;
    unsigned int soil_to_fertilizer_size;
    LocMapComponent *fertilizer_to_water;
    unsigned int fertilizer_to_water_size;
    LocMapComponent *water_to_light;
    unsigned int water_to_light_size;
    LocMapComponent *light_to_temperature;
    unsigned int light_to_temperature_size;
    LocMapComponent *temperature_to_humidity;
    unsigned int temperature_to_humidity_size;
    LocMapComponent *humidity_to_location;
    unsigned int humidity_to_location_size;
};


__device__
unsigned int get_seed(const unsigned int *seed_input, unsigned int seed_input_size, unsigned int i) {
    unsigned int seeds_seen = 0;

    for (int seed_input_idx = 0; seed_input_idx < seed_input_size; seed_input_idx += 2) {
        unsigned int range_len = seed_input[seed_input_idx + 1];
        if (i < seeds_seen + range_len) {
            unsigned int seed = seed_input[seed_input_idx] + i - seeds_seen;
            return seed;
        }
        seeds_seen += range_len;
    }

    printf("ERROR: seed not found\n");
    assert(false);
}

__device__
unsigned int get_next_loc(
    const unsigned int seed,
    const LocMapComponent *loc_arr,
    const int arr_size
) {
    unsigned int next_loc = seed;
    for (int loc_idx = 0; loc_idx < arr_size; loc_idx++) {
        LocMapComponent loc = loc_arr[loc_idx];
        if (seed >= loc.from) {
            if (seed <= loc.to) {
                next_loc = seed + loc.offset;
            }
            break;
        }
    }
    return next_loc;
}

extern "C" __global__ void calc_dest_min(
    unsigned int *local_mins,
    const Garden garden,
    const int batch_size,
    const unsigned int n
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ unsigned int block_mins[];

    unsigned int min_loc = UINT_MAX;
    for (int i = 0; i < batch_size; i++) {
        unsigned int seed_idx = i + batch_idx * batch_size;
        if (seed_idx >= n) {
            return;
        }
        unsigned int seed = get_seed(garden.seed_input, garden.seed_input_size, seed_idx);
        unsigned int next_loc = seed;

        next_loc = get_next_loc(next_loc, garden.seed_to_soil, garden.seed_to_soil_size);
        next_loc = get_next_loc(next_loc, garden.soil_to_fertilizer, garden.soil_to_fertilizer_size);
        next_loc = get_next_loc(next_loc, garden.fertilizer_to_water, garden.fertilizer_to_water_size);
        next_loc = get_next_loc(next_loc, garden.water_to_light, garden.water_to_light_size);
        next_loc = get_next_loc(next_loc, garden.light_to_temperature, garden.light_to_temperature_size);
        next_loc = get_next_loc(next_loc, garden.temperature_to_humidity, garden.temperature_to_humidity_size);
        next_loc = get_next_loc(next_loc, garden.humidity_to_location, garden.humidity_to_location_size);
        min_loc = min(min_loc, next_loc);
    }

    block_mins[threadIdx.x] = min_loc;

    __syncthreads();
    unsigned int block_min = UINT_MAX;
    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            block_min = min(block_min, block_mins[i]);
        }
        local_mins[blockIdx.x] = block_min;
    }

}
