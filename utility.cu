#include "utility.h"
#include <cassert>
#include <fstream>
#include <string>
using std::ifstream;
using std::ofstream;
using std::string;
bool parseDataFrom(int& M, const int N, int& L, int charSetSize, unsigned char*** patterns, unsigned char** text, const CustomInput& CI) {
  ifstream TF(CI.textFile, std::ios::binary);
  if (!TF.is_open())
    return false;
  int perm[256], cnt = 0;
  memset(perm, -1, sizeof(perm));
  string line, buffer;
  while (getline(TF, line)) {
    buffer += line + "\n";
  }
  L = buffer.length() - 1; // strip the last '\n'
  for (int i = 0; i < L; i++) {
    unsigned char cc = buffer[i];
    if (perm[cc] == -1)
      perm[cc] = cnt++;
    buffer[i] = perm[cc];
    // if (buffer[i] < 0) {
    //   printf("buffer[%d]=%d", i, buffer[i]);
    // }
    // assert(buffer[i] >= 0);
  }

  *text = (unsigned char*) malloc((L + 1) * sizeof(unsigned char));
  memcpy(*text, buffer.c_str(), L * sizeof(unsigned char));
  TF.close();
  ifstream PF(CI.patternsFile);
  if (!PF.is_open()) {
    free(*text);
    return false;
  }
  *patterns = (unsigned char**) malloc(N * sizeof(unsigned char*));
  int k     = 0;
  M         = 0;
  while (PF >> buffer) {
    if (M == 0)
      M = buffer.length();
    else
      assert(M == buffer.length() && "All patterns must have the same length");
    for (int i = 0; i < buffer.length(); i++) {
      unsigned char cc = buffer[i];
      if (perm[cc] == -1)
        perm[cc] = cnt++;
      buffer[i] = perm[cc];
    }
    (*patterns)[k] = (unsigned char*) malloc((M + 1) * sizeof(unsigned char));
    memcpy((*patterns)[k++], buffer.c_str(), M * sizeof(unsigned char));
  }
  //   fprintf(stderr, "(L=%d,N=%d,M=%d)", L, k, M);
  printf("(L=%d, N=%d, M=%d, actualCharSet=%d) ", L, k, M, cnt);
  assert(k == N && "The number of patterns must be equal to N");
  PF.close();
  return true;
}

void randomData(int M, int N, int L, int charSetSize, unsigned char*** patterns, unsigned char** text) {
  *patterns = (unsigned char**) malloc(N * sizeof(unsigned char*));
  for (int i = 0; i < N; i++) {
    (*patterns)[i] = (unsigned char*) malloc((M + 1) * sizeof(unsigned char));
    random_string((*patterns)[i], charSetSize, M);
  }
  *text = (unsigned char*) malloc((L + 1) * sizeof(unsigned char));
  random_string(*text, charSetSize, L);
}
void dumpACGTData(int M, int N, int L, unsigned char** patterns, unsigned char* text) {
  char map[] = {'A', 'C', 'G', 'T'};
  ofstream PF("patterns_" + std::to_string(M) + "_" + std::to_string(N) + ".txt");
  PF << N << " " << M << "\n";
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++)
      PF << map[patterns[i][j]];
    PF << "\n";
  }
  PF.close();
  ofstream TF("text_" + std::to_string(M) + "_" + std::to_string(N) + ".txt");
  TF << L << "\n";
  for (int i = 0; i < L; i++)
    TF << map[text[i]];
  TF << "\n";
  TF.close();
}