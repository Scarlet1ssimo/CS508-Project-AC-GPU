template <int charSetSize>
struct SizeTraits
{
  typedef char elementTy;
};

struct SizeTraits<2>
{
  typedef int2 elementTy;
};
