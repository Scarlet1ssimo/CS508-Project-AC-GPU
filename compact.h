#pragma once

struct int4x2_t {
    unsigned char h0:4;
    unsigned char h1:4;
};

struct int2x4_t {
    unsigned char q0:2;
    unsigned char q1:2;
    unsigned char q2:2;
    unsigned char q3:2;
};

template <int charSetSize>
struct SizeTraits
{
  typedef unsigned char elementTy;
  static const int numElement = 1;
};

#define REGISTER_4_BIT(n) \
template <> \
struct SizeTraits<n> \
{ \
  typedef int4x2_t elementTy; \
  static const int numElement = 2; \
};

#define REGISTER_2_BIT(n) \
template <> \
struct SizeTraits<n> \
{ \
  typedef int2x4_t elementTy; \
  static const int numElement = 4; \
};

REGISTER_4_BIT(16)
REGISTER_4_BIT(15)
REGISTER_4_BIT(14)
REGISTER_4_BIT(13)
REGISTER_4_BIT(12)
REGISTER_4_BIT(11)
REGISTER_4_BIT(10)
REGISTER_4_BIT(9)
REGISTER_4_BIT(8)
REGISTER_4_BIT(7)
REGISTER_4_BIT(6)
REGISTER_4_BIT(5)

REGISTER_2_BIT(4)
REGISTER_2_BIT(3)
REGISTER_2_BIT(2)
REGISTER_2_BIT(1)
