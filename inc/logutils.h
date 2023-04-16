#ifndef __LOGUTIL_H
#define __LOGUTIL_H

#include <iostream>

#include "dstimer.h"

#ifdef DO_STEP_LOG
#define STEP_LOG(expression) expression
#else
#define STEP_LOG(expression)
#endif

#ifdef DO_TIME_LOG
#define TIME_LOG(expression) expression
#else
#define TIME_LOG(expression)
#endif

#endif
