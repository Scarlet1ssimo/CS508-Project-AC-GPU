#pragma once

void ACGPUSimpleLaunch(const int* tr, const char* text, int* occur, const int M, const int L, const int charSetSize);
void ACGPUSharedMemLaunch(const int* tr, const char* text, int* occur, const int M, const int L, const int charSetSize,
                          const int trieNodeNumber);
void ACGPUCoalecedMemReadLaunch(const int* tr, const char* text, int* occur, const int M, const int L, const int charSetSize);
