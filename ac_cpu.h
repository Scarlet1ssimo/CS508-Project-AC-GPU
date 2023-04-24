#pragma once
int TrieBuildCPU(unsigned char* const* patterns, int* tr, int* idx, const int M, const int N, const int charSetSize);
void TrieReorder(int* tr, int* idx, const int N, const int trieNodeNumber, const int charSetSize);
void ACBuildCPU(int* tr, int* fail, int* postOrder, const int charSetSize);
void ACCPU(const int* tr, const unsigned char* text, int* occur, const int L, const int charSetSize);
void ACPostCPU(int* out, const int* fail, const int* postOrder, const int postOrderCnt);
