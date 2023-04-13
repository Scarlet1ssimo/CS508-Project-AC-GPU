#include <queue>
using std::queue;
void random_string(char* dst, const int charSetSize, const int len) {
  for (int i = 0; i < len; ++i)
    dst[i] = rand() % charSetSize;
  dst[len] = 0;
}
int TrieBuildCPU(char* const* patterns, int* tr, int* idx, const int M, const int N, const int charSetSize) {
  int trieNodeNumber = 1;
  for (int i = 0; i < N; i++) {
    int state = 0;
    for (int j = 0; j < M; j++) {
      int c = patterns[i][j];
      if (!tr[state * charSetSize + c])
        tr[state * charSetSize + c] = trieNodeNumber++;
      state = tr[state * charSetSize + c];
    }
    idx[i] = state;
  }
  return trieNodeNumber;
}
void ACBuildCPU(int* tr, int* fail, int* postOrder, const int charSetSize) {
  queue<int> q;
  for (int c = 0; c < charSetSize; c++) {
    int state = tr[0 * charSetSize + c];
    if (state) {
      fail[state] = 0;
      q.push(state);
    }
  }
  int postOrderCnt = 0;
  while (!q.empty()) {
    int u = q.front();
    q.pop();
    postOrder[postOrderCnt++] = u;
    for (int i = 0; i < charSetSize; i++) {
      auto& v = tr[u * charSetSize + i];
      if (v) {
        fail[v] = tr[fail[u] * charSetSize + i];
        q.push(v);
      } else
        v = tr[fail[u] * charSetSize + i];
    }
  }
  // postOrderCnt == stateCnt-1, because state 0 is not included
}
void ACCPU(const int* tr, const char* text, int* occur, const int L, const int charSetSize) {
  int state = 0;
  for (int i = 0; i < L; i++) {
    state = tr[state * charSetSize + text[i]];
    occur[state]++;
  }
}
void ACPostCPU(int* out, const int* fail, const int* postOrder, const int postOrderCnt) {
  for (int i = postOrderCnt - 1; i >= 0; i--)
    out[fail[postOrder[i]]] += out[postOrder[i]];
}
