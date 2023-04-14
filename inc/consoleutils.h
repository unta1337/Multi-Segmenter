#ifndef __CONSOLEUTILS_H
#define __CONSOLEUTILS_H

#ifdef _WIN32
#include <Windows.h>
#endif

#define INIT_CONSOLE() initConsole()
void initConsole() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
}

#endif
