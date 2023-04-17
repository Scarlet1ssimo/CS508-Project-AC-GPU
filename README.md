# Aho-Corasick Algorithm on GPU

### Overview of current status

Finished sequential CPU version and GPU baseline version. \
Issues we've encountered: Not many optimization options for now \
Next-week Plan: Implement the optimization 1 and 2. Test the performance. Come up with new optimization methods.


### Problem description:

It's a multi-pattern string matching problem. Let `CharSet` be character set, with `charSetSize` different characters. There are `N` patterns, each with length of `M`, and `text`, a long string with length of `L`. We need to find the number of occurrences of all patterns in side `text`.

For example:

```c
CharSet={0,1,2,3};
charSetSize=4;
L=10;
text="0120201012";
M=2;
N=3;
pattern[0]="01";
pattern[1]="20";
pattern[2]="30";
pattern[3]="12";

// We need to find occurrences for patterns in `text`, respectively:
occur[0]=3;
occur[1]=2;
occur[2]=0;
occur[3]=2;
```

## CPU:

~720ms on RAI

## GPU Baseline:

~9.5 ms on RAI

## Optimization 1

Let shared memory serve as GPU bin. But they are limited, so we use GPU bin to handle first 8192 bins.

~5.4 ms on RAI

### Optimization 1.1

Reorder the Trie, so that shallow node has small index, typically improves the hit rate of GPU Bin.

~5.6 ms on RAI.

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

The performance is recorded on my device (RTX 3060 Laptop @ WSL2). However two methods have no time difference tested on RAI (even reorder method is 0.X ms slower):

```
Running Aho-Corasick Algorithm on GPU... 5.289984 ms
...
Running Aho-Corasick Algorithm on GPU... 5.472256 ms
```

## Optimization 2

Use shared memory to coalescedly load global memory before use.

~7.0 ms on RAI
