struct LocMapComponentC {
    unsigned int from;
    unsigned int to;
    int offset;
};

struct LocMapSizes {
    unsigned int seed_to_soil;
    unsigned int soil_to_fertilizer;
    unsigned int fertilizer_to_water;
    unsigned int water_to_light;
    unsigned int light_to_temperature;
    unsigned int temperature_to_humidity;
    unsigned int humidity_to_location;
};

__device__
unsigned int get_seed(const unsigned int *seed_input, unsigned int i) {
    int seed_input_idx = 0;
    unsigned int seeds_seen = 0;
    while (true) {
        unsigned int range_len = seed_input[seed_input_idx + 1];
        if (i < seeds_seen + range_len) {
            unsigned int seed = seed_input[seed_input_idx] + i - seeds_seen;
            return seed;
        }
        seed_input_idx += 2;
        seeds_seen += range_len;
    }
}

__device__
unsigned int get_next_loc(
    const unsigned int seed,
    const LocMapComponentC *loc_arr,
    const int arr_size
) {
    unsigned int next_loc = seed;
    for (int loc_idx = 0; loc_idx < arr_size; loc_idx++) {
        LocMapComponentC loc = loc_arr[loc_idx];
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
    const unsigned int *seed_input,
    // maps beg
    // seed to soil
    const LocMapComponentC *seed_to_soil,
    const LocMapComponentC *soil_to_fertilizer,
    const LocMapComponentC *fertilizer_to_water,
    const LocMapComponentC *water_to_light,
    const LocMapComponentC *light_to_temperature,
    const LocMapComponentC *temperature_to_humidity,
    const LocMapComponentC *humidity_to_location,
    const LocMapSizes loc_map_sizes,
    const int batch_size,
    const unsigned int n
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int min_loc = UINT_MAX;
    for (int i = 0; i < batch_size; i++) {
        unsigned int seed_idx = i + batch_idx * batch_size;
        if (seed_idx >= n) {
            return;
        }
        unsigned int seed = get_seed(seed_input, seed_idx);
        unsigned int next_loc = seed;

        next_loc = get_next_loc(next_loc, seed_to_soil, loc_map_sizes.seed_to_soil);
        next_loc = get_next_loc(next_loc, soil_to_fertilizer, loc_map_sizes.soil_to_fertilizer);
        next_loc = get_next_loc(next_loc, fertilizer_to_water, loc_map_sizes.fertilizer_to_water);
        next_loc = get_next_loc(next_loc, water_to_light, loc_map_sizes.water_to_light);
        next_loc = get_next_loc(next_loc, light_to_temperature, loc_map_sizes.light_to_temperature);
        next_loc = get_next_loc(next_loc, temperature_to_humidity, loc_map_sizes.temperature_to_humidity);
        next_loc = get_next_loc(next_loc, humidity_to_location, loc_map_sizes.humidity_to_location);
        min_loc = min(min_loc, next_loc);

    }
    local_mins[batch_idx] = min_loc;
}
