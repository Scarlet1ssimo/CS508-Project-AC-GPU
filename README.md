## Optimization 1

Shared memory. But it's limited, so we use GPU Bin to handle first 8192 bins.

### Optimization 1.1

Reorder the Trie, so that shallow node has small node index, typically improves the hit rate of GPU Bin.

No reordering:

```
Bin portion:8192/29743=0.28
Branch portion:47808747/121874993=0.39
...
Running Aho-Corasick Algorithm on GPU... 11.395072 ms
```

With reordering

```
Bin portion:8192/29743=0.28
Branch portion:63455139/121874993=0.52
...
Running Aho-Corasick Algorithm on GPU... 10.717184 ms
```
