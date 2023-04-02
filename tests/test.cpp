#include "gtest/gtest.h"
#include "library.h"
#include "testlib.h"
#include <omp.h>

TEST(HelloTest, BasicAssertions) {
    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
    printf("Test inc/main.h %d\n", LIBRARY_MACRO(1));
    printf("Test library.cpp func1 %d\n", func1(1));
    printf("Test tests/testlib.h %d\n", TEST_MACRO(1));
    printf("Test tests/testlib.cpp %d\n", test_func1(1));

    #pragma omp parallel num_threads(10)
    printf("Current thread: %d\n", omp_get_thread_num());
}