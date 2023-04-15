#ifndef __LOGUTIL_H
#define __LOGUTIL_H

#include <iostream>

#include "dstimer.h"

#ifdef DO_STEP_LOG
void step_log(std::string message) { std::cout << message << "\n"; }
#else
void step_log(std::string message) { }
#endif

#ifdef DO_TIME_LOG
void time_log(DS_timer& timer) { timer.printTimer(); }
#else
void time_log(DS_timer& timer) { }
#endif

#endif
