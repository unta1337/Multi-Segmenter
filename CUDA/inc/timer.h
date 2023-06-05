#define TIMER_ENTRY(key, name) key,

#define TIMER_LIST \
    TIMER_ENTRY(TIMER_DATA_TRANSFER_H2D, "Data Transfer Time (Host > Device)") \
    TIMER_ENTRY(TIMER_DATA_TRANSFER_D2H, "Data Transfer Time (Device > Host)") \
    TIMER_ENTRY(TIMER_KERNEL, "Kernel") \
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
    timer.initTimers();