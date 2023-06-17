#ifndef __SEGMENTUNION_H
#define __SEGMENTUNION_H

#include <vector>

struct SegmentUnion {
    std::vector<int> segment_union;
    size_t group_count;

    SegmentUnion(size_t count, int initial_value = -1) {
        segment_union = std::vector<int>(count, initial_value);
        group_count = 0;
    }
};

#endif