#pragma once
void random_string(char* dst, const int charSetSize, const int len);
int TrieBuildCPU(char* const* patterns, int* tr, int* idx, const int M, const int N, const int charSetSize);
void ACBuildCPU(int* tr, int* fail, int* postOrder, const int charSetSize);
void ACCPU(const int* tr, const char* text, int* occur, const int L, const int charSetSize);
void ACPostCPU(int* out, const int* fail, const int* postOrder, const int postOrderCnt);