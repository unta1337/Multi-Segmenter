#include "consoleutils.h"
#include "gtest/gtest.h"

int main(int argc, char** argv) {
    INIT_CONSOLE();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
