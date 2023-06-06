#ifndef __TIMER_H
#define __TIMER_H

#define TIMER_ENTRY(key, name) key,

#define TIMER_LIST \
    TIMER_ENTRY(TIMER_PREPROCESSING, "Preprocessing") \
    TIMER_ENTRY(TIMER_NORMAL_VECTOR_COMPUTATION, "  - Normal Vector Computation") \
    TIMER_ENTRY(TIMER_MAP_COUNT, "  - Map Count") \
    TIMER_ENTRY(TIMER_NORMAL_MAP_INSERTION, "  - Normal Map Insertion") \
    TIMER_ENTRY(TIMER_CC_N_TMG, "Connectivity Checking and Triangle Mesh Generating") \
    TIMER_ENTRY(TIMER_FACEGRAPH_INIT_A, "  - FaceGraph: Init A") \
    TIMER_ENTRY(TIMER_FACEGRAPH_INIT_B, "  - FaceGraph: Init B") \
    TIMER_ENTRY(TIMER_FACEGRAPH_GET_SETMENTS_A, "  - FaceGraph: Get Segments A") \
    TIMER_ENTRY(TIMER_FACEGRAPH_GET_SETMENTS_B, "  - FaceGraph: Get Segments B") \
    TIMER_ENTRY(TIMER_TRIANGLE_MESH_GENERATING, "  - Triangle Mesh Generating") \
    TIMER_ENTRY(TIMER_SEGMENT_COLORING, "Segment Coloring") \
    TIMER_ENTRY(TIMER_TOTAL, "Total (Preprocessing + CC & TMG)") \
    TIMER_ENTRY(TIMER_LIST_SIZE, "")

enum TimerType {
    TIMER_LIST
#undef TIMER_ENTRY
};

#define TIMER_ENTRY(key, name) char* key##_NAME = (char *) name;
TIMER_LIST
#undef TIMER_ENTRY

#define TIMER_ENTRY(key, name) key##_NAME,
char* TIMER_NAME_LIST[] = {
        TIMER_LIST
#undef TIMER_ENTRY
};

#define INIT_TIMER(timer) \
    for (int i = 0; i < TIMER_LIST_SIZE; i++) \
        timer.setTimerName(i, TIMER_NAME_LIST[i]); \
    timer.initTimers()

#endif
