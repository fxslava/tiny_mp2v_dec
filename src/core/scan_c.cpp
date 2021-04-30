// Copyright � 2021 Vladislav Ovchinnikov. All rights reserved.
#include "scan.h"

uint8_t g_scan[2][64] = {
  { 0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 },
  { 0,  8,  16, 24, 1,  9,  2,  10, 17, 25, 32, 40, 48, 56, 57, 49, // alternate scan
    41, 33, 26, 18, 3,  11, 4,  12, 19, 27, 34, 42, 50, 58, 35, 43,
    51, 59, 20, 28, 5,  13, 6,  14, 21, 29, 36, 44, 52, 60, 37, 45,
    53, 61, 22, 30, 7,  15, 23, 31, 38, 46, 54, 62, 39, 47, 55, 63 } };

void inverse_alt_scan_c(int16_t QF[64], int16_t QFS[64]) {
    for (int i = 0; i < 64; i++)
        QF[g_scan[1][i]] = QFS[i];
}

void inverse_scan_c(int16_t QF[64], int16_t QFS[64]) {
    for (int i = 0; i < 64; i++)
        QF[i] = QFS[g_scan[0][i]];
}